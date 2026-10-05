#!/bin/bash -l

# MegaDepth evaluation
# Usage:
#   bash scripts/reproduce_test/megadepth.sh <config> <checkpoint>

SCRIPTPATH=$(dirname "$(readlink -f "$0")")
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "$PROJECT_DIR"

# Use the first argument as the config path, or fall back to the default.
DEFAULT_CFG="configs/megadepth_test_sgad_plus_plus_bce.py"
main_cfg_path="${1:-$DEFAULT_CFG}"

# Use the second argument as the checkpoint path, or fall back to the environment variable.
DEFAULT_CKPT="${A2PM_SGAD_PLUS_PLUS_BCE_WEIGHTS:-}"
ckpt_path="${2:-$DEFAULT_CKPT}"

if [[ -z "${ckpt_path}" ]]; then
    echo "Provide a checkpoint as argument 2 or set A2PM_SGAD_PLUS_PLUS_BCE_WEIGHTS." >&2
    exit 2
fi

profiler_name="inference"
n_gpus_per_node=1
torch_num_workers=8
batch_size=1  # Per-GPU batch size.

echo "=========================================="
echo "Config: ${main_cfg_path}"
echo "Checkpoint: ${ckpt_path}"
echo "=========================================="

python -u ./test.py \
    "${main_cfg_path}" \
    --ckpt_path="${ckpt_path}" \
    --devices=${n_gpus_per_node} --accelerator="gpu" --strategy="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} \
    --profiler_name=${profiler_name} \
    --benchmark
