#!/usr/bin/env bash

set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 IMAGE_PATH SAVE_FOLDER [ImgSAMSeg.py options]" >&2
    echo "Example: $0 image.jpg outputs --resize-width 640 --resize-height 480" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
a2pm_root="${A2PM_SGAD_ROOT:-$(cd "${script_dir}/.." && pwd)}"
cuda_id="${CUDA_ID:-0}"

img_path="$1"
save_folder="$2"
shift 2

CUDA_VISIBLE_DEVICES="${cuda_id}" python "${a2pm_root}/segmentor/ImgSAMSeg.py" \
    --img_path "${img_path}" \
    --save_folder "${save_folder}" \
    "$@"
