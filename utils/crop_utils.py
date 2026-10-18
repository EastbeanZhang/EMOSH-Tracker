import cv2
import numpy as np
import torch

from ehm_tracker.geometry import crop_and_align_image, get_bounding_box
from ehm_tracker.geometry import transform_points as transform_points

CV2_INTERP_FLAGS = cv2.INTER_CUBIC
NP_DATA_TYPE = np.float32
DTYPE = np.float32
CV2_INTERP = cv2.INTER_LINEAR


def get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2):
    """
    Args:
        bbx_xyxy: (N, 4) [x1, y1, x2, y2]
    Returns:
        bbx_xys: (N, 3) [center_x, center_y, size]
    """
    i_p2d = torch.stack([bbx_xyxy[:, [0, 1]], bbx_xyxy[:, [2, 3]]], dim=1)
    bbx_xys = get_bbx_xys(i_p2d[None], base_enlarge=base_enlarge)[0]
    return bbx_xys


def get_bbx_xys(i_j2d, bbx_ratio=[192, 256], do_augment=False, base_enlarge=1.2):
    """Args: (B, L, J, 3) [x,y,c] -> Returns: (B, L, 3)"""
    min_x = i_j2d[..., 0].min(-1)[0]
    max_x = i_j2d[..., 0].max(-1)[0]
    min_y = i_j2d[..., 1].min(-1)[0]
    max_y = i_j2d[..., 1].max(-1)[0]
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    h = max_y - min_y
    w = max_x - min_x
    if True:
        aspect_ratio = bbx_ratio[0] / (bbx_ratio[1] + 1e-07)
        mask1 = w > aspect_ratio * h
        h[mask1] = w[mask1] / (aspect_ratio + 1e-07)
        mask2 = w < aspect_ratio * h
        w[mask2] = h[mask2] * aspect_ratio
    bbx_size = torch.max(h, w) * base_enlarge
    if do_augment:
        B, L = bbx_size.shape[:2]
        device = bbx_size.device
        if True:
            scaleFactor = torch.rand((B, L), device=device) * 0.3 + 1.05
            txFactor = torch.rand((B, L), device=device) * 1.6 - 0.8
            tyFactor = torch.rand((B, L), device=device) * 1.6 - 0.8
        else:
            scaleFactor = torch.rand((B, 1), device=device) * 0.3 + 1.05
            txFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8
            tyFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8
        raw_bbx_size = bbx_size / (base_enlarge + 1e-07)
        bbx_size = raw_bbx_size * scaleFactor
        center_x += raw_bbx_size / 2 * ((scaleFactor - 1) * txFactor + 1e-07)
        center_y += raw_bbx_size / 2 * ((scaleFactor - 1) * tyFactor + 1e-07)
    return torch.stack([center_x, center_y, bbx_size], dim=-1)


def crop_and_resize_ori(img, bbx_xy, bbx_s, dst_size=256, enlarge_ratio=1.2):
    """
    Args:
        img: (H, W, 3)
        bbx_xy: (2,)
        bbx_s: scalar
    """
    hs = bbx_s * enlarge_ratio / 2
    src = np.stack([bbx_xy - hs, bbx_xy + np.array([hs, -hs]), bbx_xy]).astype(np.float32)
    dst = np.array(
        [[0, 0], [dst_size - 1, 0], [dst_size / 2 - 0.5, dst_size / 2 - 0.5]], dtype=np.float32
    )
    A = cv2.getAffineTransform(src, dst)
    img_crop = cv2.warpAffine(img, A, (dst_size, dst_size), flags=cv2.INTER_LINEAR)
    bbx_xys_final = np.array([*bbx_xy, bbx_s * enlarge_ratio])
    return (img_crop, bbx_xys_final)


def get_batch_gvhmr(imgs, bbx_xys, img_ds=1.0, img_dst_size=256):
    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]
    if True:
        gt_bbx_size_ds = gt_bbx_size * img_ds
        ds_factors = (gt_bbx_size_ds * 1.0 / img_dst_size / 2.0).numpy()
        imgs = np.stack(
            [
                cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
                for v, d in zip(imgs, ds_factors)
            ]
        )
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize_ori(
            imgs[i], gt_center[i] * img_ds, gt_bbx_size[i] * img_ds, img_dst_size, enlarge_ratio=1.0
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / (img_ds + 1e-07)
    return (imgs, bbx_xys)


def get_iou2_tensor(box1, box2):
    xmin = torch.maximum(box1[:, 0], box2[:, 0])
    ymin = torch.maximum(box1[:, 1], box2[:, 1])
    xmax = torch.minimum(box1[:, 2], box2[:, 2])
    ymax = torch.minimum(box1[:, 3], box2[:, 3])
    inter_area = torch.maximum(torch.zeros_like(xmax - xmin), xmax - xmin) * torch.maximum(
        torch.zeros_like(ymax - ymin), ymax - ymin
    )
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    iou = inter_area / (box1_area + 1e-05)
    return iou


def get_square_box_tensor(kpts, scale_factor=1.2):
    if len(kpts.shape) < 3:
        kpts = kpts[None, ...]
    x_min = kpts[..., 0:1].min(dim=-2)[0]
    y_min = kpts[..., 1:2].min(dim=-2)[0]
    x_max = kpts[..., 0:1].max(dim=-2)[0]
    y_max = kpts[..., 1:2].max(dim=-2)[0]
    widths = x_max - x_min
    heights = y_max - y_min
    max_side = torch.maximum(widths, heights) * scale_factor
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    box = torch.stack(
        [
            x_center - max_side / 2,
            y_center - max_side / 2,
            x_center + max_side / 2,
            y_center + max_side / 2,
        ],
        dim=-1,
    )
    box = box.squeeze()
    if box.dim() == 1:
        box = box[None, ...]
    return box


def parse_bbox_from_landmark_lite(points, scale=1.5, need_square=True):
    return get_bounding_box(points, padding=scale, as_square=need_square)


def crop_image_by_bbox(image, bbox, lmk=None, dsize=256, scale=1.3):
    return crop_and_align_image(
        image, bbox, points=lmk, output_size=(dsize, dsize), padding_scale=scale
    )
