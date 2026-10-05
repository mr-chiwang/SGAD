#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
a2pm_root="${A2PM_SGAD_ROOT:-$(cd "${script_dir}/.." && pwd)}"

input_folder="${MEGADEPTH_INPUT_ROOT:-${a2pm_root}/data/testdata/megadepth_test_1500/Undistorted_SfM}"
output_base_folder="${MEGADEPTH_OUTPUT_ROOT:-${a2pm_root}/assets/a2pm_bboxes/megadepth1500/1296x968}"
cuda_id="${CUDA_ID:-0}"

if [[ ! -d "${input_folder}" ]]; then
    echo "MegaDepth input directory does not exist: ${input_folder}" >&2
    exit 1
fi

counter=0
while IFS= read -r -d '' scene_folder; do
    scene_name="$(basename "${scene_folder}")"
    image_folder="${scene_folder}/images"
    [[ -d "${image_folder}" ]] || continue

    save_folder="${output_base_folder}/${scene_name}"
    mkdir -p "${save_folder}"

    while IFS= read -r -d '' img_path; do
        file_name="$(basename "${img_path}")"
        save_name="${file_name%.*}.npy"
        save_path="${save_folder}/${save_name}"
        counter=$((counter + 1))

        if [[ -f "${save_path}" ]]; then
            echo "${counter}. Skipped ${img_path} (output exists)"
            continue
        fi

        CUDA_VISIBLE_DEVICES="${cuda_id}" python "${a2pm_root}/segmentor/ImgSAMSeg.py" \
            --img_path "${img_path}" \
            --save_folder "${save_folder}" \
            --save_name "${save_name}" \
            --resize-width 1296 \
            --resize-height 968
        echo "${counter}. Processed ${img_path} -> ${save_path}"
    done < <(
        find "${image_folder}" -maxdepth 1 -type f \
            \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
               -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) \
            -print0
    )
done < <(find "${input_folder}" -mindepth 1 -maxdepth 1 -type d -print0)

echo "Completed MegaDepth bbox preprocessing for ${counter} images."
