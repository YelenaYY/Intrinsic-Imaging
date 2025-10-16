import torch
import torch.nn as nn
import torch.nn.functional as F

from decomposer import Decomposer
from shader import NeuralShader


class Composer(nn.Module):
    def __init__(self, lights_dim=4, shader_expand_dim=8):
        super().__init__()
        self.decomposer = Decomposer(lights_dim=lights_dim)
        self.shader = NeuralShader(lights_dim=lights_dim, expand_dim=shader_expand_dim)

    @torch.no_grad()
    def _make_fg_mask(self, mask):
        # mask: (B,3,H,W) values in [0,1], <0.25 = background
        # Collapse to 1-channel foreground mask
        # mask: (B,3,H,W) → fg: (B,1,H,W) booleans
        fg = (mask >= 0.25).any(dim=1, keepdim=True)
        return fg

    def forward(self, img, mask):
        """
        img:  (B,3,256,256)
        mask: (B,3,256,256) values in [0,1], <0.25 = background
        """
        reflectance, depth, normals, lights = self.decomposer(img, mask) 

        # Shader expects normals roughly unit-length; your decomposer normalized already.
        shading = self.shader(normals, lights)  # (B,1,H,W)

        # normalize the shading to [0,1]
        shading = torch.sigmoid(shading)

        # Recompose
        I_hat = reflectance * shading  # (B,3,H,W) * (B,1,H,W)

        return {
            "I_hat": I_hat,
            "reflectance": reflectance,
            "normals": normals,
            "depth": depth,
            "lights": lights,
            "shading": shading
        }