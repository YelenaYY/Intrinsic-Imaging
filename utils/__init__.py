"""
Author: Yue (Yelena) Yu, Rongfei (Eric) Jin
Purpose: Utilities package for Intrinsic Imaging
- Exports utility functions for image processing and checkpoint management
- Provides common helper functions used throughout the project
"""

from .image import normalize_normals, masked_l1, compute_shading_gt, mask_image
from .checkpoint import find_lastest_checkpoint

__all__ = ['normalize_normals', 'masked_l1', 'find_lastest_checkpoint', 'compute_shading_gt', 'mask_image']