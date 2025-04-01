#!/usr/bin/env bash
set -x

METHOD=vit_base_patchX
DSET='172_0.1'
LR=1e-7
WD=1e-4
BS=2048
EP=200
WARM_EP=20
freezeECG='False'
PY_ARGS=${@:1}

PYTHON=${PYTHON:-"python"}
CKPT_DIR=/mnt/data2/PPG/Work/P2E_v2/Signal_alignment/${METHOD}/${DSET}/ep${EP}_WARM_EP${WARM_EP}_lr${LR}_bs${BS}_wd${WD}_freezeECG${freezeECG}

mkdir -p /mnt/data2/PPG/Work/P2E_v2/Signal_alignment/
mkdir -p ${CKPT_DIR}
echo ${CKPT_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

export CUDA_VISIBLE_DEVICES=3
${PYTHON} -u main_pretrain.py \
    --output_dir ${CKPT_DIR} \
    --log_dir ${CKPT_DIR} \
    --batch_size ${BS} \
    --model ${METHOD} \
    --epochs ${EP} \
    --warmup_epochs ${WARM_EP} \
    --blr ${LR} \
    --weight_decay ${WD} \
    --device 'cuda:0'\
    --pin_mem \
    --freezeECG ${freezeECG} \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log
