import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import load_pretrained_weights

from .dapl import DAPL, PromptLearner, TextEncoder, load_clip_to_cpu


class SelectiveStateSpaceMixer(nn.Module):
    """A lightweight Mamba-style mixer for short prompt token sequences."""

    def __init__(self, dim, expand=2, conv_kernel=3, dropout=0.0):
        super().__init__()
        hidden_dim = dim * expand
        self.in_proj = nn.Linear(dim, hidden_dim)
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=hidden_dim,
        )
        self.a_proj = nn.Linear(hidden_dim, hidden_dim)
        self.b_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @autocast()
    def forward(self, x):
        # x: [batch, seq, dim]
        u = F.silu(self.in_proj(x))
        gate = torch.sigmoid(self.gate_proj(x))

        # Depth-wise local mixing before selective scan.
        u_conv = self.dwconv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        a = torch.sigmoid(self.a_proj(u_conv))
        b = self.b_proj(u_conv)

        h = torch.zeros(
            x.size(0),
            u_conv.size(-1),
            device=x.device,
            dtype=u_conv.dtype,
        )
        outputs = []

        for t in range(x.size(1)):
            h = a[:, t, :] * h + (1.0 - a[:, t, :]) * b[:, t, :]
            outputs.append(h)

        y = torch.stack(outputs, dim=1)
        y = y * gate
        y = self.out_proj(self.dropout(y))

        return y


class PromptLearnerMamba(PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        m_cfg = cfg.TRAINER.DAPL_MAMBA
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_mixer = SelectiveStateSpaceMixer(
            dim=ctx_dim,
            expand=m_cfg.MAMBA_EXPAND,
            conv_kernel=m_cfg.MAMBA_CONV_KERNEL,
            dropout=m_cfg.MAMBA_DROPOUT,
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(m_cfg.MAMBA_RESIDUAL_SCALE))
        )

    @autocast()
    def forward(self):
        prompts = super().forward()

        n_tokens = self.n_ctx + self.n_dmx
        n_prompt_rows = self.n_cls * self.n_dm

        prompt_rows = prompts[:n_prompt_rows]
        prompt_token_block = prompt_rows[:, 1 : 1 + n_tokens, :]

        mixed_tokens = self.ctx_mixer(prompt_token_block)
        scale = torch.sigmoid(self.residual_scale).type_as(mixed_tokens)
        prompt_rows = prompt_rows.clone()
        prompt_rows[:, 1 : 1 + n_tokens, :] = prompt_token_block + scale * mixed_tokens

        return torch.cat([prompt_rows, prompts[n_prompt_rows:]], dim=0)


class CustomCLIPMamba(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearnerMamba(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class DAPL_MAMBA(DAPL):
    """DAPL with a lightweight Mamba-style prompt sequence mixer."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DAPL_MAMBA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.DAPL_MAMBA.PREC in ["fp32", "amp"]:
            # CLIP default precision is fp16.
            clip_model.float()

        print("Building custom CLIP with Mamba-style prompt mixer")
        self.model = CustomCLIPMamba(cfg, classnames, clip_model)

        self.n_dm = self.model.prompt_learner.n_dm + 1
        self.n_cls = self.model.prompt_learner.n_cls

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_learner", self.model.prompt_learner, self.optim, self.sched
        )

        self.scaler = (
            GradScaler() if cfg.TRAINER.DAPL_MAMBA.PREC == "amp" else None
        )

    def forward_backward(self, batch_x, batch_u):
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.DAPL_MAMBA.PREC
        mcfg = self.cfg.TRAINER.DAPL_MAMBA

        if prec == "amp":
            with autocast():
                output_x = self.model(image_x)
                output_u = self.model(image_u)

                pseudo_label = torch.softmax(
                    output_u[:, -self.n_cls:].reshape(-1, self.n_cls) / mcfg.T,
                    dim=-1,
                )
                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(mcfg.TAU).float()

                loss_x = F.cross_entropy(output_x[:, : self.n_cls], label)
                loss_u = (
                    F.cross_entropy(
                        output_u[:, self.n_cls : 2 * self.n_cls],
                        label_p,
                        reduction="none",
                    )
                    * mask
                ).sum() / mask.sum().clamp_min(1.0)
                loss = loss_x + mcfg.U * loss_u

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_x = self.model(image_x)
            output_u = self.model(image_u)

            pseudo_label = torch.softmax(
                output_u[:, -self.n_cls:].reshape(-1, self.n_cls) / mcfg.T,
                dim=-1,
            )
            max_probs, label_p = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(mcfg.TAU).float()

            loss_x = F.cross_entropy(output_x[:, : self.n_cls], label)
            loss_u = (
                F.cross_entropy(
                    output_u[:, self.n_cls : 2 * self.n_cls],
                    label_p,
                    reduction="none",
                )
                * mask
            ).sum() / mask.sum().clamp_min(1.0)
            loss = loss_x + mcfg.U * loss_u

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "acc_x": compute_accuracy(output_x[:, : self.n_cls], label)[0].item(),
        }

        self.update_lr()

        return loss_summary
