import torch


def estimate_focal_length(img_w, img_h):
    return (img_w**2 + img_h**2) ** 0.5


def estimate_K(img_w, img_h):
    focal_length = estimate_focal_length(img_w, img_h)
    K = torch.eye(3).float()
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = img_w / 2.0
    K[1, 2] = img_h / 2.0
    return K


def compute_bbox_info_bedlam(bbx_xys, K_fullimg):
    """impl as in BEDLAM
    Args:
        bbx_xys: ((B), N, 3), in pixel space described by K_fullimg
        K_fullimg: ((B), (N), 3, 3)
    Returns:
        bbox_info: ((B), N, 3)
    """
    fl = K_fullimg[..., 0, 0].unsqueeze(-1)
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    (cx, cy, b) = (bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2])
    bbox_info = torch.stack([cx - icx, cy - icy, b], dim=-1)
    bbox_info = bbox_info / fl
    return bbox_info


def compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg):
    (s, tx, ty) = (pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2])
    focal_length = K_fullimg[..., 0, 0]
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-09)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-09)
    tz = 2 * focal_length / (sb + 1e-09)
    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t


def normalize_kp2d(obs_kp2d, bbx_xys, clamp_scale_min=False):
    """
    Args:
        obs_kp2d: (B, L, J, 3) [x, y, c]
        bbx_xys: (B, L, 3)
    Returns:
        obs: (B, L, J, 3)  [x, y, c]
    """
    obs_xy = obs_kp2d[..., :2]
    obs_conf = obs_kp2d[..., 2]
    center = bbx_xys[..., :2]
    scale = bbx_xys[..., [2]]
    xy_max = center + scale / 2
    xy_min = center - scale / 2
    invisible_mask = (
        (obs_xy[..., 0] < xy_min[..., None, 0])
        + (obs_xy[..., 0] > xy_max[..., None, 0])
        + (obs_xy[..., 1] < xy_min[..., None, 1])
        + (obs_xy[..., 1] > xy_max[..., None, 1])
    )
    obs_conf = obs_conf * ~invisible_mask
    if clamp_scale_min:
        scale = scale.clamp(min=1e-05)
    normalized_obs_xy = 2 * (obs_xy - center.unsqueeze(-2)) / scale.unsqueeze(-2)
    return torch.cat([normalized_obs_xy, obs_conf[..., None]], dim=-1)
