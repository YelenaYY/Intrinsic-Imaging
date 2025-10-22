"""
Author: Yue (Yelena) Yu, Rongfei (Eric) Jin
Purpose: Decomposer neural network model for Intrinsic Imaging
- Predicts reflectance, depth, surface normals, and lighting from input images
- Uses encoder-decoder architecture with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decomposer(nn.Module):
    """
    Predicts reflectance, depth, surface normals, and lighting from an image.
    Input: 3x256x256 images
    Outputs:
    - reflectance: same size as input
    - normals: same size as input
    - depth: same HW size as input but with 1 channel
    - lights: (lights_dim * num_lights)d vector
    """
    
    def __init__(self, lights_dim=4, num_lights=1):
        super(Decomposer, self).__init__()
        
        self.encoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16)),
            nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32)),
            nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64)),
            nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128)),
            nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256)),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        ])
        
        self.decoder_reflectance = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256)),
            nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256)),
            nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128)),
            nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64)),
            nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32)),
            nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16)),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        ])
        
        # Normals decoder
        self.decoder_normals = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256)),
            nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256)),
            nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128)),
            nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64)),
            nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32)),
            nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16)),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        ])
        
        # Depth decoder
        self.decoder_depth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256)),
            nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256)),
            nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128)),
            nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64)),
            nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32)),
            nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16)),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        ])
        
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.lights_encoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128)),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        ])
        
        lights_encoded_dim = 2
        self.lights_fc1 = nn.Linear(64 * (lights_encoded_dim ** 2), 32)
        total_lights_dim = lights_dim * num_lights
        self.lights_fc2 = nn.Linear(32, total_lights_dim)
    
    def decode_branch(self, decoder, encoded, x):
        """Decode with skip connections from encoder."""
        for idx in range(len(decoder) - 1):
            x = decoder[idx](x)
            if idx != 0:
                x = self.upsampler(x)
            # Skip connection
            x = torch.cat([x, encoded[-(idx + 1)]], dim=1)
            x = F.leaky_relu(x)
        
        x = decoder[-1](x)
        return x
    
    def forward(self, img, mask):
        """
        Inputs:
        - img: Input tensor (B, 3, H, W)
        - mask: Mask tensor (B, 3, H, W) where values < 0.25 indicate background

        Outputs:
        - reflectance: (B, 3, H, W)
        - depth: (B, 1, H, W)
        - normals: (B, 3, H, W)
        - lights: (B, lights_dim * num_lights)
        """
        # Shared encoding
        x = img
        encoded = []
        for layer in self.encoder:
            x = layer(x)
            x = F.leaky_relu(x)
            encoded.append(x)
        
        # Decode lighting parameters
        lights = x
        for layer in self.lights_encoder:
            lights = layer(lights)
            lights = F.leaky_relu(lights)
        lights = lights.view(lights.size(0), -1)
        lights = F.leaky_relu(self.lights_fc1(lights))
        lights = self.lights_fc2(lights)
        
        # Decode intrinsic images
        reflectance = self.decode_branch(self.decoder_reflectance, encoded, x)
        normals = self.decode_branch(self.decoder_normals, encoded, x)
        depth = self.decode_branch(self.decoder_depth, encoded, x)
        
        # Process normals: clamp and normalize
        rg = torch.clamp(normals[:, :2, :, :], -1, 1)
        b = torch.clamp(normals[:, 2:3, :, :], 0, 1)
        normals_clamped = torch.cat([rg, b], dim=1)
        
        # Normalize to unit vectors
        magnitude = torch.norm(normals_clamped, p=2, dim=1, keepdim=True)
        normals_normalized = normals_clamped / (magnitude + 1e-6)
        
        # Apply mask
        mask_bool = mask < 0.25
        reflectance[mask_bool] = 0
        normals_normalized[mask_bool] = 0
        depth[mask_bool[:, 0:1]] = 0
        
        return reflectance, depth, normals_normalized, lights