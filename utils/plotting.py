import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def _compute_conf_thresh(dataset_name):
    if dataset_name == 'ScanNet':
        thr = 5e-4
    elif dataset_name == 'MegaDepth':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr

import matplotlib.lines
import os

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        text=[], dpi=100, path=None, gap=10):
    # Determine the canvas size for the two input images.
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    max_h = max(h0, h1)
    total_width = w0 + w1 + gap

    # Place both images on a shared white RGB canvas.
    img_concat = np.ones((max_h, total_width, 3), dtype=np.uint8) * 255

    if len(img0.shape) == 2:
        img0_color = np.repeat(img0[:, :, np.newaxis], 3, axis=-1)
    else:
        img0_color = img0
    if len(img1.shape) == 2:
        img1_color = np.repeat(img1[:, :, np.newaxis], 3, axis=-1)
    else:
        img1_color = img1

    img_concat[:h0, :w0, :] = img0_color
    img_concat[:h1, w0 + gap:, :] = img1_color

    # Shift second-image keypoints into concatenated-canvas coordinates.
    mkpts1_shifted = mkpts1 + np.array([w0 + gap, 0])

    # Draw correspondence lines and endpoints on the combined image.
    fig, ax = plt.subplots(figsize=(15, 10), dpi=dpi)
    ax.imshow(img_concat)

    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        for i in range(len(mkpts0)):
            ax.add_line(matplotlib.lines.Line2D(
                [mkpts0[i, 0], mkpts1_shifted[i, 0]],
                [mkpts0[i, 1], mkpts1_shifted[i, 1]],
                color=color[i], linewidth=1
            ))

        ax.scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        ax.scatter(mkpts1_shifted[:, 0], mkpts1_shifted[:, 1], c=color, s=4)

    ax.axis('off')

    # Choose annotation color from the brightness of the upper-left patch.
    txt_color = 'k' if img_concat[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=ax.transAxes,
        fontsize=13, va='top', ha='left', color=txt_color)

    # Save to disk when a path is supplied; otherwise return the figure.
    if path:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig



import random

def _make_evaluation_figure(dataset_name, corrs, img0, img1, epi_errs, pose_err, alpha='dynamic', path=None):
    if len(corrs) > 5000:
        corrs = random.sample(corrs, 5000)
    corrs = np.array(corrs) # N x 4
    epi_errs = np.array(epi_errs)

    # Equalize array lengths for plotting. This cannot recover correspondence
    # identity if the caller sampled corrs and errors independently.
    if len(corrs) != len(epi_errs):
        n = min(len(corrs), len(epi_errs))
        corrs = corrs[:n]
        epi_errs = epi_errs[:n]

    if len(corrs) == 0:
        return None

    conf_thr = _compute_conf_thresh(dataset_name)

    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    
    kpts0, kpts1 = corrs[:, :2], corrs[:, 2:]
    n_correct = np.sum(correct_mask)

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
            
    err_R = pose_err[0]
    err_t = pose_err[1]
    
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'err_R: {err_R:.4f}, err_t: {err_t:.4f}'
    ]
    
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text, path=path)
    return figure

def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)
