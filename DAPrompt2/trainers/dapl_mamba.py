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


class CrossDomainGatedInteractor(nn.Module):
    """Dual-state source-target text interaction with confidence-based gates."""

    def __init__(self, dim, cfg):
        super().__init__()
        m_cfg = cfg.TRAINER.DAPL_MAMBA
        self.conf_temp = float(m_cfg.CROSS_CONF_TEMP)
        self.gate_temp = float(m_cfg.CROSS_GATE_TEMP)
        self.conf_detach = bool(m_cfg.CROSS_CONF_DETACH)

        self.state_mixer = SelectiveStateSpaceMixer(
            dim=dim,
            expand=m_cfg.CROSS_MAMBA_EXPAND,
            conv_kernel=m_cfg.CROSS_MAMBA_CONV_KERNEL,
            dropout=m_cfg.CROSS_MAMBA_DROPOUT,
        )
        self.src_scale = nn.Parameter(
            torch.tensor(float(m_cfg.CROSS_RESIDUAL_SCALE))
        )
        self.tgt_scale = nn.Parameter(
            torch.tensor(float(m_cfg.CROSS_RESIDUAL_SCALE))
        )
        self.src_proj = nn.Linear(dim, dim)
        self.tgt_proj = nn.Linear(dim, dim)

    @autocast()
    def forward(self, src_text, tgt_text, src_conf, tgt_conf, tau):
        src_w = src_conf.detach() if self.conf_detach else src_conf
        tgt_w = tgt_conf.detach() if self.conf_detach else tgt_conf

        # Use class text states for cross-domain exchange; confidence only controls gates.
        src_state = src_text.mean(dim=0)
        tgt_state = tgt_text.mean(dim=0)
        states = torch.stack([src_state, tgt_state], dim=0).unsqueeze(0)
        mixed_states = self.state_mixer(states).squeeze(0)
        src_msg = self.src_proj(mixed_states[0]).unsqueeze(0)
        tgt_msg = self.tgt_proj(mixed_states[1]).unsqueeze(0)

        src_conf_mean = src_w.mean()
        tgt_conf_mean = tgt_w.mean()
        gate_t2s = torch.sigmoid(self.gate_temp * (tgt_conf_mean - tau))
        gate_s2t = torch.sigmoid(self.gate_temp * (src_conf_mean - tau))

        src_scale = torch.sigmoid(self.src_scale).type_as(src_text)
        tgt_scale = torch.sigmoid(self.tgt_scale).type_as(tgt_text)

        src_text = src_text + gate_t2s * src_scale * src_msg
        tgt_text = tgt_text + gate_s2t * tgt_scale * tgt_msg

        src_text = src_text / src_text.norm(dim=-1, keepdim=True)
        tgt_text = tgt_text / tgt_text.norm(dim=-1, keepdim=True)

        gate_info = {
            "gate_t2s": gate_t2s,
            "gate_s2t": gate_s2t,
            "src_conf_mean": src_conf_mean,
            "tgt_conf_mean": tgt_conf_mean,
        }

        return src_text, tgt_text, gate_info


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
        self.enable_cross_interact = bool(m_cfg.CROSS_INTERACT)
        self.cross_interactor = CrossDomainGatedInteractor(ctx_dim, cfg)

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

    @autocast()
    def cross_domain_interaction(self, text_features, n_cls, src_conf, tgt_conf, tau):
        if not self.enable_cross_interact:
            gate_info = {
                "gate_t2s": text_features.new_tensor(0.0),
                "gate_s2t": text_features.new_tensor(0.0),
                "src_conf_mean": src_conf.mean(),
                "tgt_conf_mean": tgt_conf.mean(),
            }
            return text_features, gate_info

        src_text = text_features[:n_cls]
        tgt_text = text_features[n_cls : 2 * n_cls]
        remainder = text_features[2 * n_cls :]

        src_text, tgt_text, gate_info = self.cross_interactor(
            src_text, tgt_text, src_conf, tgt_conf, tau
        )

        text_features = torch.cat([src_text, tgt_text, remainder], dim=0)

        return text_features, gate_info


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
        image_features = self.encode_image(image)
        text_features = self.encode_text()
        logits = self.compute_logits(image_features, text_features)

        return logits

    @autocast()
    def encode_image(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    @autocast()
    def encode_text(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    @autocast()
    def compute_logits(self, image_features, text_features):
        return self.logit_scale.exp() * image_features @ text_features.t()

    @autocast()
    def forward_pair(self, image_x, image_u, n_cls, tau):
        text_features = self.encode_text()
        src_features = self.encode_image(image_x)
        tgt_features = self.encode_image(image_u)

        src_logits_base = self.compute_logits(src_features, text_features)
        tgt_logits_base = self.compute_logits(tgt_features, text_features)

        conf_temp = float(self.prompt_learner.cross_interactor.conf_temp)
        src_conf = torch.softmax(src_logits_base[:, :n_cls] / conf_temp, dim=-1).max(
            dim=-1
        )[0]
        tgt_conf = torch.softmax(tgt_logits_base[:, -n_cls:] / conf_temp, dim=-1).max(
            dim=-1
        )[0]

        text_features, gate_info = self.prompt_learner.cross_domain_interaction(
            text_features, n_cls, src_conf, tgt_conf, tau
        )

        src_logits = self.compute_logits(src_features, text_features)
        tgt_logits = self.compute_logits(tgt_features, text_features)

        return src_logits, tgt_logits, gate_info


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
                output_x, output_u, gate_info = self.model.forward_pair(
                    image_x, image_u, self.n_cls, mcfg.TAU
                )

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
            output_x, output_u, gate_info = self.model.forward_pair(
                image_x, image_u, self.n_cls, mcfg.TAU
            )

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
            "gate_t2s": gate_info["gate_t2s"].item(),
            "gate_s2t": gate_info["gate_s2t"].item(),
            "src_conf": gate_info["src_conf_mean"].item(),
            "tgt_conf": gate_info["tgt_conf_mean"].item(),
        }

        self.update_lr()

        return loss_summary
