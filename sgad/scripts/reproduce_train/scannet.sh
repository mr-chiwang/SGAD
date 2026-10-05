#!/bin/bash -l

# ScanNet training
# Usage:
#   bash scripts/reproduce_train/scannet.sh
#   bash scripts/reproduce_train/scannet.sh configs/scannet_sgad_plus_plus_focal.py

# Optional NCCL settings for multi-GPU training on RTX 4090 nodes.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_LEVEL=PIX
# export NCCL_SOCKET_IFNAME=lo
# export TORCH_DISTRIBUTED_TIMEOUT=3600
# export PYTHONUNBUFFERED=1

SCRIPTPATH=$(dirname "$(readlink -f "$0")")
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "$PROJECT_DIR"

# Use the first argument as the config path, or fall back to the default.
DEFAULT_CFG="configs/scannet_sgad_plus_plus_bce.py"
main_cfg_path="${1:-$DEFAULT_CFG}"

# Derive the experiment name from the config filename.
cfg_basename=$(basename "${main_cfg_path}" .py)

n_gpus=8
torch_num_workers=4
batch_size=64
pin_memory=true
exp_name="scannet-${cfg_basename}-bs=$(($n_gpus * $batch_size))"

echo "=========================================="
echo "Config: ${main_cfg_path}"
echo "Exp Name: ${exp_name}"
echo "GPUs: ${n_gpus}, Batch Size: ${batch_size}"
echo "=========================================="

python -u ./train.py "${main_cfg_path}" \
    --exp_name="${exp_name}" \
    --accelerator="gpu" --devices=${n_gpus} --precision=32 \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=20 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=100
