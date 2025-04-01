#!/usr/bin/env bash
set -x

dataset=MIMIC
METHOD=vit_base_patchX
signal='ppg'  # 'ppg' or 'ecg' or 'both' or 'enhancedppg'
DSET='172_0.0_balance'
LR=1e-6
WD=1e-4
BS=300
EP=100
WARM_EP=10
PY_ARGS=${@:1}
ecg_pretrained=None
ppg_pretrained=None
label=459
downtask_type=BCE

PYTHON=${PYTHON:-"python"}
CKPT_DIR=/mnt/data2/PPG/Work/P2E_v2/Signal_downtask_v1/${dataset}/${DSET}_${METHOD}/ep${EP}_WARM_EP${WARM_EP}_lr${LR}_bs${BS}_wd${WD}_signal${signal}_label${label}_downtask_type${downtask_type}_ecg_pretrained${ecg_pretrained}_ppg_pretrained${ppg_pretrained}

mkdir -p /mnt/data2/PPG/Work/P2E_v2/Signal_downtask_v1/
mkdir -p ${CKPT_DIR}
echo ${CKPT_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

export CUDA_VISIBLE_DEVICES=2
${PYTHON} -u main_pretrain.py \
    --output_dir ${CKPT_DIR} \
    --log_dir ${CKPT_DIR} \
    --batch_size ${BS} \
    --ppg_model ${METHOD} \
    --ecg_model ${METHOD} \
    --epochs ${EP} \
    --warmup_epochs ${WARM_EP} \
    --blr ${LR} \
    --weight_decay ${WD} \
    --device 'cuda:0'\
    --pin_mem \
    --signal ${signal} \
    --label ${label} \
    --downtask_type ${downtask_type} \
    --dataset ${dataset} \
    --balance true \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log


dataset=MIMIC
METHOD=vit_base_patchX
signal='ppg'  # 'ppg' or 'ecg' or 'both' or 'enhancedppg'
DSET='172_0.0_balance'
LR=1e-6
WD=1e-4
BS=300
EP=100
WARM_EP=10
PY_ARGS=${@:1}
ecg_pretrained=SSL
ppg_pretrained=SSL
label=459
downtask_type=BCE

PYTHON=${PYTHON:-"python"}
CKPT_DIR=/mnt/data2/PPG/Work/P2E_v2/Signal_downtask_v1/${dataset}/${DSET}_${METHOD}/ep${EP}_WARM_EP${WARM_EP}_lr${LR}_bs${BS}_wd${WD}_signal${signal}_label${label}_downtask_type${downtask_type}_ecg_pretrained${ecg_pretrained}_ppg_pretrained${ppg_pretrained}

mkdir -p /mnt/data2/PPG/Work/P2E_v2/Signal_downtask_v1/
mkdir -p ${CKPT_DIR}
echo ${CKPT_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

export CUDA_VISIBLE_DEVICES=2
${PYTHON} -u main_pretrain.py \
    --output_dir ${CKPT_DIR} \
    --log_dir ${CKPT_DIR} \
    --batch_size ${BS} \
    --ppg_model ${METHOD} \
    --ecg_model ${METHOD} \
    --epochs ${EP} \
    --warmup_epochs ${WARM_EP} \
    --blr ${LR} \
    --weight_decay ${WD} \
    --device 'cuda:0'\
    --pin_mem \
    --ecg_pretrained ${ecg_pretrained} \
    --ppg_pretrained ${ppg_pretrained} \
    --signal ${signal} \
    --label ${label} \
    --downtask_type ${downtask_type} \
    --dataset ${dataset} \
    --balance true \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log





dataset=MIMIC
METHOD=vit_base_patchX
signal='ppg'  # 'ppg' or 'ecg' or 'both' or 'enhancedppg'
DSET='172_0.0_balance'
LR=1e-6
WD=1e-4
BS=300
EP=100
WARM_EP=10
PY_ARGS=${@:1}
ecg_pretrained=Align
ppg_pretrained=Align
label=459
downtask_type=BCE

PYTHON=${PYTHON:-"python"}
CKPT_DIR=/mnt/data2/PPG/Work/P2E_v2/Signal_downtask_v1/${dataset}/${DSET}_${METHOD}/ep${EP}_WARM_EP${WARM_EP}_lr${LR}_bs${BS}_wd${WD}_signal${signal}_label${label}_downtask_type${downtask_type}_ecg_pretrained${ecg_pretrained}_ppg_pretrained${ppg_pretrained}

mkdir -p /mnt/data2/PPG/Work/P2E_v2/Signal_downtask_v1/
mkdir -p ${CKPT_DIR}
echo ${CKPT_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

export CUDA_VISIBLE_DEVICES=2
${PYTHON} -u main_pretrain.py \
    --output_dir ${CKPT_DIR} \
    --log_dir ${CKPT_DIR} \
    --batch_size ${BS} \
    --ppg_model ${METHOD} \
    --ecg_model ${METHOD} \
    --epochs ${EP} \
    --warmup_epochs ${WARM_EP} \
    --blr ${LR} \
    --weight_decay ${WD} \
    --device 'cuda:0'\
    --pin_mem \
    --ecg_pretrained ${ecg_pretrained} \
    --ppg_pretrained ${ppg_pretrained} \
    --signal ${signal} \
    --label ${label} \
    --downtask_type ${downtask_type} \
    --dataset ${dataset} \
    --balance true \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log



dataset=MIMIC
METHOD=vit_base_patchX
signal='ecg'  # 'ppg' or 'ecg' or 'both' or 'enhancedppg'
DSET='172_0.0_balance'
LR=1e-6
WD=1e-4
BS=300
EP=100
WARM_EP=10
PY_ARGS=${@:1}
ecg_pretrained=None
ppg_pretrained=None
label=459
downtask_type=BCE

PYTHON=${PYTHON:-"python"}
CKPT_DIR=/mnt/data2/PPG/Work/P2E_v2/Signal_downtask_v1/${dataset}/${DSET}_${METHOD}/ep${EP}_WARM_EP${WARM_EP}_lr${LR}_bs${BS}_wd${WD}_signal${signal}_label${label}_downtask_type${downtask_type}_ecg_pretrained${ecg_pretrained}_ppg_pretrained${ppg_pretrained}

mkdir -p /mnt/data2/PPG/Work/P2E_v2/Signal_downtask_v1/
mkdir -p ${CKPT_DIR}
echo ${CKPT_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

export CUDA_VISIBLE_DEVICES=2
${PYTHON} -u main_pretrain.py \
    --output_dir ${CKPT_DIR} \
    --log_dir ${CKPT_DIR} \
    --batch_size ${BS} \
    --ppg_model ${METHOD} \
    --ecg_model ${METHOD} \
    --epochs ${EP} \
    --warmup_epochs ${WARM_EP} \
    --blr ${LR} \
    --weight_decay ${WD} \
    --device 'cuda:0'\
    --pin_mem \
    --signal ${signal} \
    --label ${label} \
    --downtask_type ${downtask_type} \
    --dataset ${dataset} \
    --balance true \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log


dataset=MIMIC
METHOD=vit_base_patchX
signal='ecg'  # 'ppg' or 'ecg' or 'both' or 'enhancedppg'
DSET='172_0.0_balance'
LR=1e-6
WD=1e-4
BS=300
EP=100
WARM_EP=10
PY_ARGS=${@:1}
ecg_pretrained=SSL
ppg_pretrained=SSL
label=459
downtask_type=BCE

PYTHON=${PYTHON:-"python"}
CKPT_DIR=/mnt/data2/PPG/Work/P2E_v2/Signal_downtask_v1/${dataset}/${DSET}_${METHOD}/ep${EP}_WARM_EP${WARM_EP}_lr${LR}_bs${BS}_wd${WD}_signal${signal}_label${label}_downtask_type${downtask_type}_ecg_pretrained${ecg_pretrained}_ppg_pretrained${ppg_pretrained}

mkdir -p /mnt/data2/PPG/Work/P2E_v2/Signal_downtask_v1/
mkdir -p ${CKPT_DIR}
echo ${CKPT_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

export CUDA_VISIBLE_DEVICES=2
${PYTHON} -u main_pretrain.py \
    --output_dir ${CKPT_DIR} \
    --log_dir ${CKPT_DIR} \
    --batch_size ${BS} \
    --ppg_model ${METHOD} \
    --ecg_model ${METHOD} \
    --epochs ${EP} \
    --warmup_epochs ${WARM_EP} \
    --blr ${LR} \
    --weight_decay ${WD} \
    --device 'cuda:0'\
    --pin_mem \
    --ecg_pretrained ${ecg_pretrained} \
    --ppg_pretrained ${ppg_pretrained} \
    --signal ${signal} \
    --label ${label} \
    --downtask_type ${downtask_type} \
    --dataset ${dataset} \
    --balance true \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log





dataset=MIMIC
METHOD=vit_base_patchX
signal='ecg'  # 'ppg' or 'ecg' or 'both' or 'enhancedppg'
DSET='172_0.0_balance'
LR=1e-6
WD=1e-4
BS=300
EP=100
WARM_EP=10
PY_ARGS=${@:1}
ecg_pretrained=Align
ppg_pretrained=Align
label=459
downtask_type=BCE

PYTHON=${PYTHON:-"python"}
CKPT_DIR=/mnt/data2/PPG/Work/P2E_v2/Signal_downtask_v1/${dataset}/${DSET}_${METHOD}/ep${EP}_WARM_EP${WARM_EP}_lr${LR}_bs${BS}_wd${WD}_signal${signal}_label${label}_downtask_type${downtask_type}_ecg_pretrained${ecg_pretrained}_ppg_pretrained${ppg_pretrained}

mkdir -p /mnt/data2/PPG/Work/P2E_v2/Signal_downtask_v1/
mkdir -p ${CKPT_DIR}
echo ${CKPT_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

export CUDA_VISIBLE_DEVICES=2
${PYTHON} -u main_pretrain.py \
    --output_dir ${CKPT_DIR} \
    --log_dir ${CKPT_DIR} \
    --batch_size ${BS} \
    --ppg_model ${METHOD} \
    --ecg_model ${METHOD} \
    --epochs ${EP} \
    --warmup_epochs ${WARM_EP} \
    --blr ${LR} \
    --weight_decay ${WD} \
    --device 'cuda:0'\
    --pin_mem \
    --ecg_pretrained ${ecg_pretrained} \
    --ppg_pretrained ${ppg_pretrained} \
    --signal ${signal} \
    --label ${label} \
    --downtask_type ${downtask_type} \
    --dataset ${dataset} \
    --balance true \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log


