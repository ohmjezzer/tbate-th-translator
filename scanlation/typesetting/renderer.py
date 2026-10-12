# -*- coding: utf-8 -*-
"""
typesetting/renderer.py
=======================
ตัววาดข้อความลงภาพด้วย Pillow (PIL)

จุดสำคัญเพื่อ "เนียนตา" (professional look):
    1. จัดกึ่งกลางตาม bounding box จริงของข้อความ (measure ก่อนวาด)
    2. Auto-contrast : พื้นมืด→ตัวอักษรขาว, พื้นสว่าง→ตัวอักษรดำ
    3. Stroke สำหรับ SFX ให้อ่านชัดบนภาพที่ซับซ้อน
    4. Clip-to-mask : ตัดข้อความไม่ให้ล้นออกนอกบอลลูน (ใช้ mask จาก SAM)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .base import TextLayout, TextStyle


def luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


class PillowRenderer:
    def __init__(self, auto_contrast: bool = True):
        self.auto_contrast = auto_contrast

    # ---- วัดความสว่างพื้นหลังของบอลลูน ----------------------------------
    @staticmethod
    def background_luminance(image: np.ndarray, box: Tuple[int, int, int, int],
                             mask: Optional[np.ndarray] = None) -> float:
        x0, y0, x1, y1 = box
        region = image[y0:y1, x0:x1]
        if region.size == 0:
            return 255.0
        if mask is not None and mask.shape == image.shape[:2]:
            m = mask[y0:y1, x0:x1]
            if m.any():
                pixels = region[m]
            else:
                pixels = region.reshape(-1, 3)
        else:
            pixels = region.reshape(-1, 3)
        # ใช้ median เพื่อไม่ให้ตัวอักษรเดิมที่อาจตกค้างมาหลอกค่า
        med = np.median(pixels, axis=0)
        return luminance((float(med[0]), float(med[1]), float(med[2])))

    def resolve_color(self, style: TextStyle, bg_lum: float) -> Tuple[int, int, int]:
        """เลือกสีตัวอักษร (auto contrast)"""
        if style.color is not None:
            return style.color
        if not self.auto_contrast:
            return (20, 20, 20)
        return (245, 245, 245) if bg_lum < 128 else (18, 18, 18)

    def resolve_stroke(self, style: TextStyle, text_color: Tuple[int, int, int]
                       ) -> Optional[Tuple[int, int, int]]:
        if style.stroke_width <= 0:
            return None
        if style.stroke_fill is not None:
            return style.stroke_fill
        # ขอบตัดกับสีตัวอักษร (ขาว↔ดำ)
        return (0, 0, 0) if luminance(text_color) > 128 else (255, 255, 255)

    # ---- วาดข้อความ ------------------------------------------------------
    def draw_layout(self, target: Image.Image, layout: TextLayout, style: TextStyle,
                    box: Tuple[int, int, int, int], bg_lum: float) -> None:
        """วาด layout ลงบน target (PIL.Image)"""
        if not layout.lines:
            return
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2
        color = self.resolve_color(style, bg_lum)
        stroke = self.resolve_stroke(style, color)

        draw = ImageDraw.Draw(target)
        total_h = layout.line_height * len(layout.lines)
        ty = (y0 + y1) / 2 - total_h / 2

        for line in layout.lines:
            lw = draw.textlength(line, font=layout.font)
            if style.align == "left":
                tx = x0
            elif style.align == "right":
                tx = x1 - lw
            else:  # center
                tx = cx - lw / 2
            draw.text(
                (tx, ty), line, font=layout.font, fill=color,
                stroke_width=style.stroke_width,
                stroke_fill=stroke,
            )
            ty += layout.line_height

    # ---- composite แบบ clip ด้วย mask -----------------------------------
    @staticmethod
    def composite_clipped(base: Image.Image, layer: Image.Image,
                          mask: np.ndarray, feather: int = 1) -> Image.Image:
        """
        รวม layer (RGBA, ข้อความที่วาดแล้ว) เข้ากับ base โดยให้ข้อความโผล่เฉพาะ
        บริเวณ mask (บอลลูน) — กันข้อความล้นออกนอกบอลลูน
        """
        import cv2

        if mask.sum() == 0:
            return base
        m = (mask.astype(np.uint8) * 255)
        if feather > 0:
            m = cv2.erode(m, np.ones((2 * feather + 1,) * 2, np.uint8))
        alpha = layer.split()[-1]
        alpha = Image.fromarray(np.minimum(np.asarray(alpha), m))
        layer = layer.copy()
        layer.putalpha(alpha)
        return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
