#!/bin/bash

# custom config
DATA=/home/dails26/X/mywork/data # 你的 office-home 数据集所在的父目录

DATASET=$1 # name of the dataset
CFG=$2  # config file
T=$3 # temperature
TAU=$4 # pseudo label threshold
U=$5 # coefficient for loss_u
NAME=$6 # job name
TRAINER=${7:-DAPL}

if [ "${TRAINER}" = "DAPL_MAMBA" ]; then
    CFG_PREFIX=TRAINER.DAPL_MAMBA
else
    CFG_PREFIX=TRAINER.DAPL
fi


for SEED in 1 
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        ${CFG_PREFIX}.T ${T} \
        ${CFG_PREFIX}.TAU ${TAU} \
        ${CFG_PREFIX}.U ${U}
    fi
done
