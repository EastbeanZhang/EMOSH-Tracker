from typing import Dict

import torch
from torch import nn

from .backbones import create_backbone
from .heads import build_mano_head


class HAMER(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = create_backbone(cfg)
        self.mano_head = build_mano_head(cfg)

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """
        x = batch["img"]
        batch_size = x.shape[0]
        conditioning_feats = self.backbone(x[:, :, :, 32:-32])
        (pred_mano_params, pred_cam, _) = self.mano_head(conditioning_feats)
        output = {}
        output["pred_cam"] = pred_cam
        output["pred_mano_params"] = {k: v.clone() for (k, v) in pred_mano_params.items()}
        device = pred_mano_params["hand_pose"].device
        dtype = pred_mano_params["hand_pose"].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(
            batch_size, 2, device=device, dtype=dtype
        )
        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-09),
            ],
            dim=-1,
        )
        output["pred_cam_t"] = pred_cam_t
        output["focal_length"] = focal_length
        pred_mano_params["global_orient"] = pred_mano_params["global_orient"].reshape(
            batch_size, -1, 3, 3
        )
        pred_mano_params["hand_pose"] = pred_mano_params["hand_pose"].reshape(batch_size, -1, 3, 3)
        pred_mano_params["betas"] = pred_mano_params["betas"].reshape(batch_size, -1)
        return output
