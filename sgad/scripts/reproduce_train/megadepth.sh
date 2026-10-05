#!/bin/bash -l

# MegaDepth training
# Usage:
#   bash scripts/reproduce_train/megadepth.sh
#   bash scripts/reproduce_train/megadepth.sh configs/megadepth_train_sgad_plus_plus_focal.py

SCRIPTPATH=$(dirname "$(readlink -f "$0")")
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "$PROJECT_DIR"

# Use the first argument as the config path, or fall back to the default.
DEFAULT_CFG="configs/megadepth_train_sgad_plus_plus_bce.py"
main_cfg_path="${1:-$DEFAULT_CFG}"

# Derive the experiment name from the config filename.
cfg_basename=$(basename "${main_cfg_path}" .py)

n_gpus=8
torch_num_workers=2
batch_size=32
pin_memory=true
exp_name="megadepth-${cfg_basename}-bs=$(($n_gpus * $batch_size))"

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
    --num_sanity_val_steps=5 \
    --benchmark=True \
    --max_epochs=100
