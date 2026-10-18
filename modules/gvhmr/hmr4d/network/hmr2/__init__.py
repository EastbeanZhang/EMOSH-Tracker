from pathlib import Path as Path

import torch

from ehm_tracker.assets import asset_path as asset_path

from .configs import get_config as get_config
from .hmr2 import HMR2 as HMR2

HMR2A_CKPT = asset_path("gvhmr/checkpoints/hmr2/epoch=10-step=25000.ckpt")


def load_hmr2(checkpoint_path=HMR2A_CKPT):
    model_cfg = str((Path(__file__).parent / "configs/model_config.yaml").resolve())
    model_cfg = get_config(model_cfg)
    model_cfg.defrost()
    model_cfg.SMPL.MEAN_PARAMS = str(asset_path("gvhmr/checkpoints/hmr2/smpl_mean_params.npz"))
    model_cfg.freeze()
    if model_cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in model_cfg.MODEL:
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, (
            f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        )
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    model = HMR2(model_cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    keys = [k for k in state_dict.keys() if k.split(".")[0] in ["backbone", "smpl_head"]]
    state_dict = {k: v for (k, v) in state_dict.items() if k in keys}
    model.load_state_dict(state_dict, strict=True)
    return model
