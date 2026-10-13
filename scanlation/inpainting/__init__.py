# -*- coding: utf-8 -*-
"""สเตจ Inpainting: ลบข้อความต้นฉบับออกจากภาพ"""

from .base import Inpainter
from .masks import build_text_mask, build_bubble_mask, combine_masks
from .lama_inpainter import LamaInpainter, IOPaintLaMaInpainter
from .lama_torch_inpainter import LamaTorchInpainter
from .opencv_inpainter import OpenCVInpainter, MedianFillInpainter

__all__ = [
    "Inpainter",
    "build_text_mask",
    "build_bubble_mask",
    "combine_masks",
    "LamaInpainter",
    "IOPaintLaMaInpainter",
    "LamaTorchInpainter",
    "OpenCVInpainter",
    "MedianFillInpainter",
]
