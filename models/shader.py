"""
Author: Yue (Yelena) Yu, Rongfei (Eric) Jin
Purpose: Shader neural network model for Intrinsic Imaging
- Generates shading from surface normals and lighting parameters
- Uses convolutional blocks with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv_block(in_ch, out_ch, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
    )


def _build_encoder(channels, kernel_size, padding, strides):
    layers = []
    for idx in range(1, len(channels)):
        in_ch = channels[idx - 1]
        out_ch = channels[idx]
        stride = strides[idx - 1]
        layers.append(_conv_block(in_ch, out_ch, kernel_size, padding, stride))
    return nn.ModuleList(layers)


class NeuralShader(nn.Module):
    """
    Predicts shading image given shape (normals) and lighting conditions.
    Output: single-channel shading with same HxW as input.
    Lights: 4D vector [x, y, z, energy] per light source.
    For multiple lights: [light1_energy, light1_x, light1_y, light1_z, light2_energy, ...]
    """

    def __init__(self, lights_dim=4, expand_dim=8, num_lights=1):
        super(NeuralShader, self).__init__()
        self.num_lights = num_lights

        # Encoder
        channels = [3, 16, 32, 64, 128, 256, 256]
        kernel_size = 3
        padding = 1
        strides = [1, 2, 2, 2, 2, 2]

        # Encoder over normals
        self.encoder = _build_encoder(channels, kernel_size, padding, strides)

        # Keep track of encoder stage output channels for skip connections
        encoder_out_channels = channels[1:]  # [16, 32, 64, 128, 256, 256]

        # Decoder:
        # Mirror links + light map at the bottleneck:
        # - append last channel size to "link" encoder/decoder depth
        # - set output (head) channels to 1 (shading)
        # - add +1 channel at bottleneck for the light map concat
        dec_channels = channels.copy()
        dec_channels.append(dec_channels[-1])  # link
        dec_channels[0] = 1  # single channel shading output
        dec_channels[-1] += 1  # concat lights map channel
        dec_channels = list(reversed(dec_channels))

        # Build decoder with stride=1 at every stage
        dec_strides = [1] * (len(dec_channels) - 1)
        self.decoder = _build_encoder(dec_channels, kernel_size, padding, dec_strides)
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')

        # After each concat with skip features, reduce channels back to the
        # expected input channels for the next decoder conv using 1x1 convs
        skip_chs = list(reversed(encoder_out_channels))[: len(self.decoder) - 1]
        self.concat_reducers = nn.ModuleList([
            nn.Conv2d(dec_channels[i + 1] + skip_chs[i], dec_channels[i + 1], kernel_size=1)
            for i in range(len(self.decoder) - 1)
        ])

        # Lights encoder to spatial map
        # For multiple lights, we process all lights together
        self.expand_dim = expand_dim
        total_lights_dim = lights_dim * num_lights
        self.lights_fc = nn.Linear(total_lights_dim, expand_dim * expand_dim)

    def forward(self, normals: torch.Tensor, lights: torch.Tensor) -> torch.Tensor:
        x = normals
        encoded = []
        for ind in range(len(self.encoder)):
            x = self.encoder[ind](x)
            x = F.leaky_relu(x)
            encoded.append(x)

        lights_map = self.lights_fc(lights)
        lights_map = lights_map.view(-1, 1, self.expand_dim, self.expand_dim)
        # Resize light map to match bottleneck spatial size
        lights_map = F.interpolate(lights_map, size=encoded[-1].shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat((encoded[-1], lights_map), dim=1)

        for ind in range(len(self.decoder) - 1):
            x = self.decoder[ind](x)
            if ind != 0:
                x = self.upsampler(x)
            x = torch.cat((x, encoded[-(ind + 1)]), dim=1)
            # Reduce concatenated channels to the expected input channels for the next conv
            x = self.concat_reducers[ind](x)
            x = F.leaky_relu(x)

        x = self.decoder[-1](x)
        return x
