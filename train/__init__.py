"""
Author: Yue (Yelena) Yu, Rongfei (Eric) Jin
Purpose: Training package for Intrinsic Imaging
- Exports all training classes for model training
- Provides unified access to model trainers
"""

from .train_decomposer import DecomposerTrainer
from .train_shader import ShaderTrainer
from .train_composer import ComposerTrainer

__all__ = ['DecomposerTrainer', 'ShaderTrainer', 'ComposerTrainer']