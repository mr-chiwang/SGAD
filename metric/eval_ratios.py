import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import os

import numpy as np
from loguru import logger

from metric.Evaluation import AMEval, PoseAUCEval, TimeEval, MMAEval

DEFAULT_RES_ROOT = REPO_ROOT / "res"


def _phi_tokens_from_ratios_dir(
    ratios_dir: Path,
    *,
    challenger_name: str,
    err_postfix: str = "pose_errs.txt",
) -> list[str]:
    if not ratios_dir.exists():
        return []

    prefix = f"{challenger_name}-"
    suffix = f"_{err_postfix}"

    tokens: list[str] = []
    for p in ratios_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        token = name[len(prefix):-len(suffix)]
        if token:
            tokens.append(token)

    def _sort_key(t: str):
        try:
            return (0, float(t))
        except Exception:
            return (1, t)

    return sorted(set(tokens), key=_sort_key)


def _resolve_phi_list(
    phis_arg: list[str],
    *,
    ratios_dir: Path,
    challenger_name: str,
    err_postfix: str = "pose_errs.txt",
) -> list[str]:
    raw_items: list[str] = []
    for item in (phis_arg or ["auto"]):
        if item is None:
            continue
        item = item.strip()
        if not item:
            continue
        if "," in item:
            raw_items.extend([x.strip() for x in item.split(",") if x.strip()])
        else:
            raw_items.append(item)

    if len(raw_items) == 0:
        raw_items = ["auto"]

    available = _phi_tokens_from_ratios_dir(
        ratios_dir, challenger_name=challenger_name, err_postfix=err_postfix
    )

    if len(raw_items) == 1 and raw_items[0].lower() == "auto":
        if not available:
            raise ValueError(
                f"No phi candidates found under {ratios_dir} for challenger '{challenger_name}'. "
                f"Expected files like '{challenger_name}-<phi>_{err_postfix}'."
            )
        return available

    resolved: list[str] = []
    for raw in raw_items:
        if raw.lower() == "auto":
            raise ValueError("Cannot mix 'auto' with explicit phi values.")

        if raw in available:
            resolved.append(raw)
            continue

        try:
            raw_val = float(raw)
        except Exception:
            raw_val = None

        if raw_val is not None and available:
            matches: list[str] = []
            for t in available:
                try:
                    if abs(float(t) - raw_val) <= 1e-9:
                        matches.append(t)
                except Exception:
                    continue
            if len(matches) == 1:
                resolved.append(matches[0])
                continue
            if len(matches) > 1:
                raise ValueError(f"Ambiguous phi={raw}; matches multiple tokens: {matches}")

        expected = ratios_dir / f"{challenger_name}-{raw}_{err_postfix}"
        if not expected.exists():
            raise ValueError(
                f"Phi '{raw}' not found. Missing {expected}. Available phis: {available}"
            )
        resolved.append(raw)

    return resolved


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pose/area metrics from ratios files.")
    p.add_argument(
        "--res-root",
        default=None,
        help="Result root directory (default: $A2PM_RES_ROOT or repo ./res).",
    )
    p.add_argument(
        "--res-folder",
        default=None,
        help="Specific result folder under res-root (default: $A2PM_RES_FOLDER).",
    )
    p.add_argument(
        "--phis",
        nargs="*",
        default=["auto"],
        help="Phi list: '--phis 2.5 6.0' or '--phis 2.5,6.0'. Use '--phis auto' to detect (default).",
    )
    p.add_argument(
        "--ameval",
        choices=["auto", "on", "off"],
        default="auto",
        help="Run AMEval: on/off/auto (auto skips if required files missing). Default: auto.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    root_path = Path(
        args.res_root or os.environ.get("A2PM_RES_ROOT", str(DEFAULT_RES_ROOT))
    ).expanduser()
    root_folder = args.res_folder or os.environ.get("A2PM_RES_FOLDER")
    if not root_folder:
        raise SystemExit(
            "Missing result folder. Provide '--res-folder <name>' or set env 'A2PM_RES_FOLDER'."
        )

    baseline_name = "pm"
    challenger_name = "a2pm"
    folder_name = "ratios"

    log_folder = root_path / root_folder
    os.makedirs(os.path.join(str(log_folder), "AMEval"), exist_ok=True)
    logger.add(
        f"{log_folder}/AMEval/res.log",
        rotation="500 MB",
        level="INFO",
        retention="10 days",
    )

    ratios_dir = log_folder / folder_name
    phis = _resolve_phi_list(
        args.phis,
        ratios_dir=ratios_dir,
        challenger_name=challenger_name,
        err_postfix="pose_errs.txt",
    )
    logger.info(f"Using res_root={root_path} res_folder={root_folder} phis={phis}")

    # Pose Error Eval
    try:
        output_path = os.path.join(str(root_path), root_folder, folder_name)

        pose_eval_cfg = {
            "root_path": os.path.join(str(root_path), root_folder),
            "folder_name": folder_name,
            "baseline_name": baseline_name,
            "challenger_name": challenger_name,
            "phi_list": phis,
            "output_path": output_path,
        }

        pose_eval = PoseAUCEval(pose_eval_cfg)
        pose_eval.run()
    except Exception as e:
        logger.error(f"PoseAUCEval failed: {e}")

    # AMEval
    try:
        am_name = "am"
        required = [
            ratios_dir / f"{am_name}_acr.txt",
            ratios_dir / f"{am_name}_iou.txt",
            ratios_dir / f"{am_name}_aor.txt",
        ]

        run_ameval = args.ameval == "on"
        if args.ameval == "auto":
            run_ameval = all(p.exists() for p in required)
            if not run_ameval:
                missing = [str(p) for p in required if not p.exists()]
                logger.info(f"Skipping AMEval (missing files): {missing}")

        if run_ameval:
            am_eval_cfg = {
                "root_path": os.path.join(str(root_path), root_folder),
                "name": am_name,
                "aor_thresholds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "iou_thresholds": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            }

            am_eval = AMEval(am_eval_cfg)
            am_eval.run()
    except Exception as e:
        logger.error(f"AMEval failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
