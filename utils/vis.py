import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import random
import copy

import os
from loguru import logger

from .img_process import img_to_color

def draw_matched_area(img0, img1, area0, area1, color, out_path, name0, name1, save=True):
    """
    """
    img0 = copy.deepcopy(img0)
    img1 = copy.deepcopy(img1)
    
    if len(img0.shape) == 2:
        img0 = img_to_color(img0)
    if len(img1.shape) == 2:
        img1 = img_to_color(img1)
    
    W, H = img0.shape[1], img0.shape[0]
    
    out = stack_img(img0, img1)

    draw_matched_area_in_img(out, area0, area1, color)

    if save:
        cv2.imwrite(os.path.join(out_path, f"{name0}_{name1}_matched_area.png"), out)
        logger.info(f"save matched area to {os.path.join(out_path, f'{name0}_{name1}_matched_area.png')}")
    
    return out

def draw_matched_area_list(img0, img1, area0_list, area1_list, out_path, name0, name1, save=True):
    """
    """
    n = len(area0_list)
    assert n == len(area1_list)

    color_map = get_n_colors(n)

    if len(img0.shape) == 2:
        img0 = img_to_color(img0)
    if len(img1.shape) == 2:  
        img1 = img_to_color(img1)

    W, H = img0.shape[1], img0.shape[0]
    out = stack_img(img0, img1)

    flag = True
    for i in range(n):
        color = color_map[i]
        flag = draw_matched_area_in_img(out, area0_list[i], area1_list[i], color)
    
    if save:
        cv2.imwrite(os.path.join(out_path, f"{name0}_{name1}_matched_areas.png"), out)
    
    return flag

def paint_semantic(ins0, ins1, out_path="", name0="", name1="", save=True):
    """ fill color by sematic label
    """
    assert len(ins0.shape) == len(ins1.shape) == 2

    H, W = ins0.shape

    label_list = []
    label_color_dict = {}

    for i in range(H):
        for j in range(W):
            temp0 = ins0[i, j]
            temp1 = ins1[i, j]
            if temp0 not in label_list:
                label_list.append(temp0)
            if temp1 not in label_list:
                label_list.append(temp1)

    label_list = sorted(label_list)


    N = len(label_list)
    cmaps_='gist_ncar'
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, N)))

    for i in range(N):
        # black is background
        if label_list[i] == 0:
            label_color_dict[label_list[i]] = [0, 0, 0]
            continue
        c = cmap(i)
        label_color_dict[label_list[i]] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]

    

    outImg0 = np.zeros((H,W,3))

    for i in range(H):
        for j in range(W):
            outImg0[i, j, :] = label_color_dict[ins0[i,j]]

    outImg1 = np.zeros((H,W,3))

    for i in range(H):
        for j in range(W):
            outImg1[i, j, :] = label_color_dict[ins1[i,j]]

    if save:
        cv2.imwrite(os.path.join(out_path, "{0}_color.jpg".format(name0)), outImg0)
        cv2.imwrite(os.path.join(out_path, "{0}_color.jpg".format(name1)), outImg1)

    return outImg0, outImg1

def get_n_colors(n):
    """
    """
    label_color_dict = {}

    cmaps_='gist_ncar'
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, n)))

    for i in range(n):
        c = cmap(i)
        label_color_dict[i] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
    
    return label_color_dict

def stack_img(img0, img1):
    """ stack two image in horizontal
    Args:
        img0: numpy array 3 channel
    """
    # assert img0.shape == img1.shape

    if len(img0.shape) == 2:
        img0 = img_to_color(img0)
    if len(img1.shape) == 2:    
        img1 = img_to_color(img1)

    assert len(img0.shape) == 3

    W0, H0 = img0.shape[1], img0.shape[0]
    W1, H1 = img1.shape[1], img1.shape[0]

    H_s = max(H0, H1)
    W_s = W0 + W1

    out = 255 * np.ones((H_s, W_s, 3), np.uint8)

    try:
        out[:H0, :W0, :] = img0.copy()
        out[:H1, W0:, :] = img1.copy()
    except ValueError as e:
        logger.exception(e)
        logger.info(f"img0 shape is {img0.shape}, img1 shape is {img1.shape}")
        logger.info(f"out shape is {out.shape}")
        raise e

    return out

def draw_matched_area_in_img(out, patch0, patch1, color):
    """
    """
    W = out.shape[1] // 2
    W = int(W)
    patch0 = [int(i) for i in patch0]
    patch1_s = [patch1[0]+W, patch1[1]+W, patch1[2], patch1[3]]
    try:
        patch1_s = [int(i) for i in patch1_s]
    except ValueError as e:
        logger.exception(e)
        return False



    cv2.rectangle(out, (patch0[0], patch0[2]), (patch0[1], patch0[3]), tuple(color), 3)
    try:
        cv2.rectangle(out, (patch1_s[0], patch1_s[2]), (patch1_s[1], patch1_s[3]), color, 3)
    except cv2.error:
        logger.exception("what?")
        return False

    line_s = [(patch0[0]+patch0[1])//2, (patch0[2]+patch0[3])//2]
    line_e = [(patch1_s[0]+patch1_s[1])//2, (patch1_s[2]+patch1_s[3])//2]

    cv2.line(out, (line_s[0], line_s[1]), (line_e[0], line_e[1]), color=color, thickness=3, lineType=cv2.LINE_AA)

    return True

import numpy as np
import cv2
import os

def plot_matches_lists_lr(image0, image1, matches, outPath, name, num_samples=500):
    """
    Args:
    matches: [u0, v0, u1, v1]s
    num_samples: number of points to sample and visualize
    """
    image0 = img_to_color(image0)
    image1 = img_to_color(image1)
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]
    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0:, :] = image1

    # Uniformly sample specified number of matches
    if len(matches) > num_samples:
        indices = np.linspace(0, len(matches) - 1, num_samples, dtype=int)
        matches = [matches[i] for i in indices]

    color = np.zeros((len(matches), 3), dtype=int)
    color[:, 1] = 255

    for match, c in zip(matches, color):
        c = c.tolist()
        u0, v0, u1, v1 = match
        u0, v0 = int(u0), int(v0)
        u1, v1 = int(u1) + W0, int(v1)
        
        cv2.line(out, (u0, v0), (u1, v1), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u0, v0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u1, v1), 2, c, -1, lineType=cv2.LINE_AA)

    path = os.path.join(outPath, name + ".jpg")
    logger.info(f"save match list img to {path}")
    cv2.imwrite(path, out)
    
def plot_matches_lists_ud(image0, image1, matches, outPath, name):
    """
    Args:
        matches: [u0, v0, u1,v1]s
    """
    image0 = img_to_color(image0)
    image1 = img_to_color(image1)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = H0 + H1, max(W0, W1)
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[H0:, :W1, :] = image1

    color = np.zeros((len(matches), 3), dtype=int)
    color[:, 1] = 255

    for match, c in zip(matches, color):
        c = c.tolist()
        u0, v0, u1, v1 = match
        u0 = int(u0)
        v0 = int(v0)
        u1 = int(u1) 
        v1 = int(v1) + H0
        cv2.line(out, (u0, v0), (u1, v1), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u0, v0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u1, v1), 2, c, -1, lineType=cv2.LINE_AA)

    path = os.path.join(outPath, name+".jpg")
    logger.info(f"save match list img to {path}")
    cv2.imwrite(path, out)

def plot_matches_with_mask_ud(image0, image1, mask, matches, outPath, name, sample_num=500):
    """
    Args:
        mask: 0 -> false match
    """
    # random sample
    if len(matches) > sample_num:
        matches = random.sample(matches, sample_num)

    image0 = img_to_color(image0)
    image1 = img_to_color(image1)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = H0 + H1, max(W0, W1)
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[H0:, :W1, :] = image1

    for i, match in enumerate(matches):
        if mask[i] == 0: c = [0, 0, 255]
        if mask[i] == 1: c = [0, 255, 0]
        if mask[i] == -1: continue

        u0, v0, u1, v1 = match
        u0 = int(u0)
        v0 = int(v0)
        u1 = int(u1) 
        v1 = int(v1) + H0
        cv2.line(out, (u0, v0), (u1, v1), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u0, v0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u1, v1), 2, c, -1, lineType=cv2.LINE_AA)


    path = os.path.join(outPath, name+".jpg")
    logger.info(f"save match list img to {path}")
    cv2.imwrite(path, out)

    return out

import matplotlib.patches as patches

def plot_bbox_matches(data, indices_area0, indices_area1, save_dir, draw_points=False, num_points=50):
    """Render matched bounding boxes and optionally their point matches.

    Args:
        data: Dictionary containing images and bounding boxes.
        indices_area0: Matched region indices for the first image.
        indices_area1: Matched region indices for the second image.
        save_dir: Directory for rendered image pairs.
        draw_points: Whether to draw point correspondences when available.
        num_points: Maximum number of point correspondences drawn per pair.
    """
    # Create the output directory before rendering any pairs.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Read original image dimensions retained in the model input dictionary.
    raw_h_0, raw_w_0 = data['image_shape0'][0], data['image_shape0'][1]
    raw_h_1, raw_w_1 = data['image_shape1'][0], data['image_shape1'][1]
    
    # Use the prepared images stored in the input dictionary.
    # image0 = data['image0'][:, :raw_h_0, :raw_w_0].detach().cpu().numpy()
    # image1 = data['image1'][:, :raw_h_1, :raw_w_1].detach().cpu().numpy()
    
    image0 = data['image0']
    image1 = data['image1']
    
    # Remove a leading singleton channel dimension from grayscale images.
    if image0.ndim == 3 and image0.shape[0] == 1:
        image0 = image0.squeeze(0)  # (1, h, w) -> (h, w)
    if image1.ndim == 3 and image1.shape[0] == 1:
        image1 = image1.squeeze(0)
    
    # Convert channel-first RGB arrays to the layout expected by imshow.
    if image0.ndim == 3 and image0.shape[0] == 3:
        image0 = np.transpose(image0, (1, 2, 0))
    if image1.ndim == 3 and image1.shape[0] == 3:
        image1 = np.transpose(image1, (1, 2, 0))
    
    # Convert byte-range images to floating-point display range when needed.
    if image0.max() > 1.0:
        image0 = image0 / 255.0
    if image1.max() > 1.0:
        image1 = image1 / 255.0
    
    # Pair the supplied region indices.
    indices_img0 = indices_area0
    indices_img1 = indices_area1
    points_matches = []

    # Bounding boxes are represented as [x, y, width, height].
    bboxes0 = data['bboxes0']
    bboxes1 = data['bboxes1']
    bboxes0_format = 'xywh'
    bboxes1_format = 'xywh'
    
    # Render one figure for each matched region pair.
    for i, (img0_idx, img1_idx) in enumerate(zip(indices_img0, indices_img1)):
        # Place the images side by side without inter-panel spacing.
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={'wspace': 0, 'hspace': 0})

        # Use an explicit grayscale colormap for single-channel images.
        if image0.ndim == 3:
            ax0.imshow(image0)
        else:
            ax0.imshow(image0, cmap='gray')

        if image1.ndim == 3:
            ax1.imshow(image1)
        else:
            ax1.imshow(image1, cmap='gray')

        # Resolve the two matched boxes from their indices.
        bbox0 = bboxes0[img0_idx].cpu().numpy()
        bbox1 = bboxes1[img1_idx].cpu().numpy()

        def draw_bbox(ax, bbox, color, bbox_format):
            """Draw a bounding box and return its center coordinates.

            Args:
                ax: Matplotlib axes receiving the box.
                bbox: Bounding-box coordinates.
                color: Box color.
                bbox_format: Either ``xywh`` or flattened polygon ``corners``.
            """
            if bbox_format == 'xywh':
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                         linewidth=3, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                return center
            elif bbox_format == 'corners':
                polygon = patches.Polygon(bbox.reshape(-1, 2), closed=True, linewidth=3, edgecolor=color, facecolor='none')
                ax.add_patch(polygon)
                center = np.mean(bbox.reshape(-1, 2), axis=0)
                return center
            else:
                raise ValueError("Unsupported bbox format")

        # Draw both boxes and connect their centers across the two axes.
        center0 = draw_bbox(ax0, bbox0, 'r', bboxes0_format)
        center1 = draw_bbox(ax1, bbox1, 'r', bboxes1_format)

        con = patches.ConnectionPatch(xyA=center0, xyB=center1, coordsA="data", coordsB="data",
                                      axesA=ax0, axesB=ax1, color='r', linewidth=2, arrowstyle="->")
        ax1.add_artist(con)

        # Label each box with its source-region index.
        ax0.text(center0[0], center0[1], f'{img0_idx}', color='white', fontsize=12,
                 bbox=dict(facecolor='red', edgecolor='none', alpha=0.5, pad=0))
        ax1.text(center1[0], center1[1], f'{img1_idx}', color='white', fontsize=12,
                 bbox=dict(facecolor='red', edgecolor='none', alpha=0.5, pad=0))

        # Draw point matches only when a corresponding point set is available.
        if draw_points and i < len(points_matches):
            matched_points = np.array(points_matches[i])
            points_img0_full, points_img1_full = matched_points[:, :2], matched_points[:, 2:]
            num_available = len(points_img0_full)
            if num_available > 0:
                num_to_draw = min(num_points, num_available)
                # Select evenly spaced entries from the available point matches.
                indices = np.linspace(0, num_available - 1, num_to_draw, dtype=int)
                sampled_points_img0 = [points_img0_full[idx] for idx in indices]
                sampled_points_img1 = [points_img1_full[idx] for idx in indices]

                for p0, p1 in zip(sampled_points_img0, sampled_points_img1):
                    ax0.plot(p0[0], p0[1], 'bo', markersize=5)
                    ax1.plot(p1[0], p1[1], 'bo', markersize=5)
                    # Connect corresponding points across the image panels.
                    con_point = patches.ConnectionPatch(xyA=(p0[0], p0[1]), xyB=(p1[0], p1[1]),
                                                        coordsA="data", coordsB="data",
                                                        axesA=ax0, axesB=ax1, color='blue', linewidth=1, linestyle='--')
                    ax1.add_artist(con_point)

        # Remove axes and outer padding from the rendered figure.
        ax0.axis('off')
        ax1.axis('off')

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.tight_layout()

        # Persist this matched pair and release the Matplotlib figure.
        save_path = os.path.join(save_dir, f'matched_pair_{i}.png')
        fig.savefig(save_path)

        plt.close(fig)
        
        logger.info(f"save matched pair {i} to {save_path}")

    logger.info(f"save all matched pairs to {save_dir}")
