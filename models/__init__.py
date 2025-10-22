"""
Author: Yue (Yelena) Yu, Rongfei (Eric) Jin
Purpose: Models package for Intrinsic Imaging
- Exports all neural network models used in the project
- Provides unified access to decomposer, shader, and composer models
"""

from .decomposer import Decomposer
from .shader import NeuralShader
from .shader_variant import NeuralShaderVariant
from .composer import Composer

__all__ = ['Decomposer', 'NeuralShader', 'NeuralShaderVariant', 'Composer']
