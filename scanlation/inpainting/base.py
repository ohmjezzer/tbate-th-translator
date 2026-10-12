# -*- coding: utf-8 -*-
"""
inpainting/base.py
==================
Interface ของสเตจ Inpainting

สัญญา (contract) ง่าย ๆ:
    image (H x W x 3 RGB) + mask (H x W bool/uint8, True = จุดที่ต้องลบ)
        → image ใหม่ที่ลบข้อความแล้ว

ทุกตัว implement เหมือนกัน → สลับโมเดลได้โดยไม่แตะ pipeline
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Inpainter(ABC):
    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """image: RGB uint8, mask: bool/uint8 (ค่า !=0 = พื้นที่ที่ต้องลบ)"""
        raise NotImplementedError

    # helper ใช้ร่วม: normalize mask ให้เป็น uint8 0/255
    @staticmethod
    def _to_u8(mask: np.ndarray) -> np.ndarray:
        if mask.dtype == np.uint8:
            return (mask > 0).astype(np.uint8) * 255
        return (mask.astype(bool)).astype(np.uint8) * 255
