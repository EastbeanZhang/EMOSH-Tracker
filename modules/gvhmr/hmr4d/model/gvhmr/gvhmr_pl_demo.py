import torch
from torch import nn

from ....hmr4d.configs import MainStore, builds
from ....hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import Pipeline
from ....hmr4d.utils.geo.hmr_cam import normalize_kp2d
from ....hmr4d.utils.pylogger import Log


class DemoPL(nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        # self.pipeline = instantiate(pipeline, _recursive_=False)

        self.pipeline = Pipeline(pipeline.args, pipeline.args_denoiser3d)

    @torch.no_grad()
    def predict(self, data, static_cam=False):
        """auto add batch dim
        data: {
            "length": int, or Torch.Tensor,
            "kp2d": (F, 3)
            "bbx_xys": (F, 3)
            "K_fullimg": (F, 3, 3)
            "cam_angvel": (F, 3)
            "f_imgseq": (F, 3, 256, 256)
        }

        """

        batch = {
            "length": data["length"][None],
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
            "bbx_xys": data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "f_imgseq": data["f_imgseq"][None],
        }
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = self.pipeline.forward(batch, train=False, postproc=True, static_cam=static_cam)

        pred = {
            "smpl_params_global": {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()},
            "smpl_params_incam": {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()},
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,  # intermediate outputs
        }
        return pred

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"Loading GVHMR checkpoint: {ckpt_path}")

        state_dict = torch.load(ckpt_path, "cpu", weights_only=False)["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        learned = set(dict(self.named_parameters()))
        missing_weights = sorted(learned.intersection(missing))
        if missing_weights:
            raise RuntimeError(f"GVHMR checkpoint is missing learned weights: {missing_weights}")
        if unexpected:
            Log.warning("Ignored unused checkpoint keys: %s", unexpected)


MainStore.store(
    name="gvhmr_pl_demo",
    node=builds(DemoPL, pipeline="${pipeline}"),
    group="model/gvhmr",
)
