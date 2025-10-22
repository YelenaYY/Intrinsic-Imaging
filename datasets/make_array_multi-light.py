"""
Author: Yue (Yelena) Yu, Rongfei (Eric) Jin
Purpose: Multi-light array generation script
- Generates light parameter arrays for multi-light training scenarios
- Creates numpy arrays with specified number of light sources
"""

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_lights', type=int, default=2, help='Number of light sources')
parser.add_argument('--lights_energy_low', type=float, default=1.0)
parser.add_argument('--lights_energy_high', type=float, default=5.0)
parser.add_argument('--lights_pos_low', nargs=3, type=float, default=[-5.0, -2.5, -1.0], help='x y z lower bounds')
parser.add_argument('--lights_pos_high', nargs=3, type=float, default=[5.0, 3.5, 5.0], help='x y z upper bounds')
parser.add_argument('--size', type=int, default=20000, help='Number of samples to generate')
parser.add_argument('--save_path', default='datasets/arrays/shader_multilights.npy')
args = parser.parse_args()

def random(low, high):
    """Generate random value(s) between low and high bounds"""
    if type(high) == list:
        params = [np.random.uniform(low=low[ind], high=high[ind]) for ind in range(len(high))]
    else:
        params = np.random.uniform(low=low, high=high)
    return params

def generate_light_params():
    """Generate parameters for a single light: [energy, x, y, z]"""
    low = [args.lights_energy_low] + args.lights_pos_low
    high = [args.lights_energy_high] + args.lights_pos_high
    return random(low, high)

# Generate parameters for multiple lights
# Shape: (size, num_lights * 4) where each light has [energy, x, y, z]
all_params = []
for i in range(args.size):
    sample_params = []
    for light_idx in range(args.num_lights):
        light_params = generate_light_params()
        sample_params.extend(light_params)
    all_params.append(sample_params)

all_params = np.array(all_params, dtype=np.float32)

print(f"Generated light parameters:")
print(f"  Number of samples: {args.size}")
print(f"  Number of lights per sample: {args.num_lights}")
print(f"  Parameters per light: 4 (energy, x, y, z)")
print(f"  Output shape: {all_params.shape}")
print(f"  Saving to: {args.save_path}")

np.save(args.save_path, all_params)