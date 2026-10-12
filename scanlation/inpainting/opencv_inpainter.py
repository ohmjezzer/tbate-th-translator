# -*- coding: utf-8 -*-
"""
inpainting/opencv_inpainter.py
==============================
ทางเลือกเบา/สำรอง สำหรับพื้นที่ที่ไม่ซับซ้อน (บอลลูนสีพื้นเรียบ)

    - OpenCVInpainter     : cv2.inpaint (Telea / Navier-Stokes) — เหมือนโค้ดเดิม
    - MedianFillInpainter : เติมด้วยสี median ของพื้นที่โดยรอบ — เร็วมาก
                            และได้ผลดีกับบอลลูนสีขาว/สีพื้นทึบ
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import InpaintingConfig
from ..utils.registry import register
from .base import Inpainter


@register("inpainter", "opencv")
class OpenCVInpainter(Inpainter):
    """
    cv2.inpaint — เป็น fallback ที่ไม่มี dependency หนัก
    เลือกอัลกอริทึมผ่าน cfg.opencv_method: "telea" | "ns"
    """

    def __init__(self, config: Optional[InpaintingConfig] = None):
        self.cfg = config or InpaintingConfig()
        self.method = (
            cv2.INPAINT_NS if self.cfg.opencv_method.lower() == "ns" else cv2.INPAINT_TELEA
        )

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.sum() == 0:
            return image.copy()
        # cv2.inpaint ต้องการภาพ BGR หรือ RGB ก็ได้ (ทำงานแยกช่องสี)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out = cv2.inpaint(bgr, self._to_u8(mask), self.cfg.opencv_radius, self.method)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


@register("inpainter", "median")
class MedianFillInpainter(Inpainter):
    """
    เติมพื้นที่ mask ด้วยสี median ที่คำนวณจาก "วงแหวนรอบ ๆ" mask
    เร็วสุดและเหมาะกับบอลลูนสีพื้นเรียบ (พื้นขาวในมังงะส่วนใหญ่)
    """

    def __init__(self, config: Optional[InpaintingConfig] = None, ring: int = 6):
        self.cfg = config or InpaintingConfig()
        self.ring = ring

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.sum() == 0:
            return image.copy()
        m = self._to_u8(mask)
        # หาสี median จากบริเวณรอบ mask: dilate mask แล้วเอาส่วนที่อยู่นอก mask เดิม
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.ring + 1,) * 2)
        ring = cv2.dilate(m, kernel) - m
        pixels = image[ring > 0]
        color = np.median(pixels, axis=0) if pixels.size else np.array([255, 255, 255])
        out = image.copy()
        out[m > 0] = color.astype(np.uint8)
        return out
