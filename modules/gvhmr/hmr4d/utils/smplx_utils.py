from .body_model.smplx_lite import SmplxLiteV437Coco17


def make_smplx(type="supermotion_v437coco17", **kwargs):
    if type != "supermotion_v437coco17":
        raise ValueError(f"Unsupported body model: {type}")
    return SmplxLiteV437Coco17(**kwargs)
