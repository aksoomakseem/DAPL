#!/bin/bash

# custom config
DATA=/home/dails26/X/mywork/data # 你的 office-home 数据集所在的父目录

DATASET=$1
CFG=$2  # config file
T=$3 # temperature
TAU=$4 # pseudo label threshold
U=$5 # coefficient for loss_u
NAME=$6 # job name
TRAINER=${7:-DAPL}


for SEED in 1
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --model-dir ${DIR} \
    --output-dir ${DIR} \
    --eval-only
done