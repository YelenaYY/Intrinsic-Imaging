import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposer import Decomposer
from .shader import NeuralShader
from .shader_variant import NeuralShaderVariant


class Composer(nn.Module):
    def __init__(self, shader, decomposer):
        super().__init__()
        self.shader = shader
        self.decomposer = decomposer

    # @torch.no_grad()
    # def _make_fg_mask(self, mask):
    #     # mask: (B,3,H,W) values in [0,1], <0.25 = background
    #     # Collapse to 1-channel foreground mask
    #     # mask: (B,3,H,W) → fg: (B,1,H,W) booleans
    #     fg = (mask >= 0.25).any(dim=1, keepdim=True)
    #     return fg

    def forward(self, img, mask):
        """
        img:  (B,3,256,256)
        mask: (B,3,256,256) values in [0,1], <0.25 = background
        """
        reflectance, depth, normals, lights = self.decomposer(img, mask) 

        # Shader expects normals roughly unit-length; your decomposer normalized already.
        shading = self.shader(normals, lights)  # (B,1,H,W)

        # normalize the shading to [0,1]
        # shading = torch.sigmoid(shading)

        # explicitly repeat the shading to 3 channels
        shading = shading.repeat(1, 3, 1, 1)

        # Recompose
        reconstructed = reflectance * shading  # (B,3,H,W) * (B,3,H,W)

        return {
            "reconstructed": reconstructed,
            "reflectance": reflectance,
            "normals": normals,
            "depth": depth,
            "lights": lights,
            "shading": shading
        }