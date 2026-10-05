import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def _draw_bbox(ax, bbox, color, alpha, bbox_format):
    if bbox_format == "xywh":
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=3, edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
    elif bbox_format == "corners":
        polygon = patches.Polygon(bbox, closed=True, linewidth=3, edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(polygon)
        return np.mean(bbox, axis=0)
    else:
        raise ValueError("Unsupported bbox format")


def _image_tensor_to_numpy(image):
    image = image.detach().cpu().numpy()

    # Remove a singleton grayscale channel.
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)

    # Convert RGB tensors from CHW to HWC layout for Matplotlib.
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    # Convert byte-like intensity ranges to floating point in [0, 1].
    if image.max() > 1.0:
        image = image / 255.0

    return image


def _show_image_pair(ax0, ax1, image0, image1):
    if image0.shape[0] == 3:
        ax0.imshow(image0)
        ax1.imshow(image1)
    else:
        ax0.imshow(image0, cmap="gray")
        ax1.imshow(image1, cmap="gray")


def bboxes_matches_top1(data):
    raw_h_0, raw_w_0 = data["image_shape0"][0][0].item(), data["image_shape0"][0][1].item()
    raw_h_1, raw_w_1 = data["image_shape1"][0][0].item(), data["image_shape1"][0][1].item()
    image0 = data["image0"][0][:, :raw_h_0, :raw_w_0]
    image1 = data["image1"][0][:, :raw_h_1, :raw_w_1]

    bboxes0_format = data["bboxes0_format"]
    bboxes1_format = data["bboxes1_format"]

    matches = data["bboxes_matches"]
    bboxes0_matched = matches["bboxes0_matched"][0]
    bboxes1_matched = matches["bboxes1_matched"][0]
    true_match_scores = matches["bboxes_match_scores"][0]
    pred_match_scores = matches["bboxes_pred_match_scores"][0]

    if len(bboxes0_matched) == 0:
        return None

    true_match_scores = torch.stack(true_match_scores)
    pred_match_scores = torch.stack(pred_match_scores)
    bboxes0_matched = torch.stack(bboxes0_matched)
    bboxes1_matched = torch.stack(bboxes1_matched)

    image0 = _image_tensor_to_numpy(image0)
    image1 = _image_tensor_to_numpy(image1)

    pred_match_scores = pred_match_scores.detach().cpu().numpy()
    true_match_scores = true_match_scores.detach().cpu().numpy()
    bboxes0_matched = bboxes0_matched.detach().cpu().numpy()
    bboxes1_matched = bboxes1_matched.detach().cpu().numpy()

    confidence_order = np.argsort(-pred_match_scores, kind="stable")
    pred_match_scores = pred_match_scores[confidence_order]
    true_match_scores = true_match_scores[confidence_order]
    bboxes0_matched = bboxes0_matched[confidence_order]
    bboxes1_matched = bboxes1_matched[confidence_order]

    # Create side-by-side axes with no inter-panel spacing.
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={"wspace": 0, "hspace": 0})

    _show_image_pair(ax0, ax1, image0, image1)

    # k = min(10, len(pred_match_scores))  # Plot at most ten matches.
    k = min(5, len(pred_match_scores))  # Use stable descending predicted-score order.

    colors = ["#FF0000", "#00FF00", "#0000FF", "#FF00FF", "#FFFF00", "#00FFFF", "#FFA500", "#800080", "#008000", "#000080"][:k]

    legend_texts = []

    for rank, (bbox0, bbox1, color) in enumerate(zip(bboxes0_matched[:k], bboxes1_matched[:k], colors)):
        pred_score = pred_match_scores[rank]
        true_score = true_match_scores[rank]

        center0 = _draw_bbox(ax0, bbox0, color, 0.3, bboxes0_format)
        center1 = _draw_bbox(ax1, bbox1, color, 0.3, bboxes1_format)

        label = f"{rank + 1}"
        ax0.text(center0[0], center0[1], label, color=color, fontsize=15, bbox=dict(facecolor="white", edgecolor="none", alpha=0.3, pad=0))
        ax1.text(center1[0], center1[1], label, color=color, fontsize=15, bbox=dict(facecolor="white", edgecolor="none", alpha=0.3, pad=0))

        con = patches.ConnectionPatch(xyA=center0, xyB=center1, coordsA="data", coordsB="data", axesA=ax0, axesB=ax1, color=color, linewidth=1)
        ax1.add_artist(con)

        legend_texts.append(f"{rank + 1}: pred_score: {pred_score:.4f}, true_score: {true_score:.4f}")

    # Place the score legend in the upper-left corner of the first image.
    ax0.text(
        0.02,
        0.98,
        "\n".join(legend_texts),
        transform=ax0.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
    )

    ax0.axis("off")
    ax1.axis("off")

    # Remove outer margins while keeping text visible.
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout()

    return fig


def make_bbox_matching_figures(data):
    """Build logger figure groups for confidence-ranked region matches.

    The ``top1`` group contains one plot with up to five matches. ``top3`` is
    retained as an empty group for the current logger interface.
    """
    return {"top1": [bboxes_matches_top1(data)], "top3": []}


__all__ = [
    "bboxes_matches_top1",
    "make_bbox_matching_figures",
]
