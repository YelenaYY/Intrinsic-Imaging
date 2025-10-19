import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv_block(in_ch, out_ch, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
    )

def _build_encoder(channels, kernel_size, padding, strides, mult=1):
    layers = []
    for idx in range(1, len(channels)):
        m = 1 if idx == 1 else mult
        in_ch = channels[idx - 1] * m
        out_ch = channels[idx]
        stride = strides[idx - 1]
        
        # Last layer: plain Conv2d without BatchNorm (matching Shader)
        if idx < len(channels) - 1:
            layers.append(_conv_block(in_ch, out_ch, kernel_size, padding, stride))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding))
    
    return nn.ModuleList(layers)

class NeuralShaderVariant(nn.Module):
    """
    Predicts shading image given shape (normals) and lighting conditions.
    Output: single-channel shading with same HxW as input.
    Lights: 4D vector [x, y, z, energy].
    """
    def __init__(self, lights_dim=4, expand_dim=8):
        super(NeuralShaderVariant, self).__init__()
        
        # Encoder
        channels = [3, 16, 32, 64, 128, 256, 256]
        kernel_size = 3
        padding = 1
        strides = [1, 2, 2, 2, 2, 2]
        
        # Encoder over normals
        self.encoder = _build_encoder(channels, kernel_size, padding, strides)
        
        # Decoder:
        # Mirror links + light map at the bottleneck
        dec_channels = channels.copy()
        dec_channels.append(dec_channels[-1])  # link encoder and decoder
        dec_channels[0] = 1  # single channel shading output
        dec_channels[-1] += 1  # add a channel for the lighting
        dec_channels = list(reversed(dec_channels))
        
        # Build decoder with stride=1 at every stage, mult=2 for skip connections
        dec_strides = [1] * (len(dec_channels) - 1)
        self.decoder = _build_encoder(dec_channels, kernel_size, padding, dec_strides, mult=2)
        
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Lights encoder to spatial map
        self.expand_dim = expand_dim
        self.lights_fc = nn.Linear(lights_dim, expand_dim * expand_dim)
    
    def forward(self, normals: torch.Tensor, lights: torch.Tensor) -> torch.Tensor:
        x = normals
        
        # Encode
        encoded = []
        for ind in range(len(self.encoder)):
            x = self.encoder[ind](x)
            x = F.leaky_relu(x)
            encoded.append(x)
        
        # Expand lights to spatial map
        lights_map = self.lights_fc(lights)
        lights_map = lights_map.view(-1, 1, self.expand_dim, self.expand_dim)
        
        # Concatenate shape and lights representations at bottleneck
        x = torch.cat((encoded[-1], lights_map), dim=1)
        
        # Decode with skip connections
        for ind in range(len(self.decoder) - 1):
            x = self.decoder[ind](x)
            if ind != 0:
                x = self.upsampler(x)
            # Concatenate with skip connection from encoder
            x = torch.cat((x, encoded[-(ind + 1)]), dim=1)
            x = F.leaky_relu(x)
        
        # Final decoder layer
        x = self.decoder[-1](x)
        
        return x