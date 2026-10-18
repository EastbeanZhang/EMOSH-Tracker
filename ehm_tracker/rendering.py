"""Screen-space mesh rendering with the same camera convention as the optimizer."""

import numpy as np
import torch


class MeshRenderer:
    def __init__(self, faces):
        self.faces = faces

    @torch.no_grad()
    def render_mesh(self, vertices, camera, background=0.0):
        from pytorch3d.renderer import (
            HardPhongShader,
            MeshRasterizer,
            PerspectiveCameras,
            PointLights,
            RasterizationSettings,
            TexturesVertex,
        )
        from pytorch3d.renderer import MeshRenderer as P3DRenderer
        from pytorch3d.structures import Meshes

        batch = len(vertices)
        (width, height) = camera["image_size"]
        transform = camera["w2c_RT"]
        rotation = transform[:, :3, :3].clone()
        translation = transform[:, :3, 3].clone()
        # OpenCV uses +x right and +y down; PyTorch3D uses the opposite screen axes.
        rotation[:, :2] *= -1
        translation[:, :2] *= -1
        principal = (
            torch.as_tensor(camera["princpt"], device=vertices.device, dtype=vertices.dtype)
            .reshape(-1, 2)
            .expand(batch, -1)
        )
        cameras = PerspectiveCameras(
            device=vertices.device,
            R=rotation.transpose(1, 2),
            T=translation,
            focal_length=camera["focal_length"],
            principal_point=principal,
            in_ndc=False,
            image_size=[(height, width)] * batch,
        )
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=(height, width),
                # Dense projections can overflow coarse bins and silently lose faces.
                bin_size=0,
                blur_radius=0.0,
                faces_per_pixel=1,
                perspective_correct=True,
            ),
        )
        lights = PointLights(device=vertices.device, location=[[0.0, 0.0, -10.0]])
        renderer = P3DRenderer(
            rasterizer=rasterizer,
            shader=HardPhongShader(device=vertices.device, cameras=cameras, lights=lights),
        )
        textures = TexturesVertex(verts_features=torch.full_like(vertices, 0.7))
        meshes = Meshes(
            verts=vertices, faces=self.faces[None].expand(batch, -1, -1), textures=textures
        )
        rgba = renderer(meshes)
        rgb = rgba[..., :3] * rgba[..., 3:] + background * (1 - rgba[..., 3:])
        return (rgb.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
