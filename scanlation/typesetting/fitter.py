# -*- coding: utf-8 -*-
"""
typesetting/fitter.py
=====================
ปรับขนาดฟอนต์อัตโนมัติ (Auto font-size fitting) ให้ข้อความพอดีบอลลูน

อัลกอริทึม: binary search บน "ขนาดฟอนต์"
    - ขนาดที่ยอมรับได้ = ข้อความ wrap แล้ว "จำนวนบรรทัด x ความสูงบรรทัด" ไม่เกิน
      ความสูงกล่อง และไม่มีคำใดกว้างเกินความกว้างกล่อง
    - ใช้ 2 รอบ (strict → relaxed): รอบแรกบังคับว่าทุกคำต้องไม่ล้น,
      รอบที่สองยอมผ่อนปรนเพื่อความครอบคลุม

ทำไมไม่วน for-loop จาก max ลงมา: เพื่อความเร็ว ใช้ binary search + cache ฟอนต์
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from .base import TextLayout
from .thai_wrapper import ThaiTextWrapper

# canvas ขนาดจิ๋วสำหรับวัดความกว้างข้อความ (Pillow ต้องมี draw object ในการวัด)
_MEASURE = ImageDraw.Draw(Image.new("RGB", (8, 8)))


class FontFitter:
    def __init__(self, wrapper: Optional[ThaiTextWrapper] = None):
        self.wrapper = wrapper or ThaiTextWrapper()
        self._font_cache: Dict[tuple, ImageFont.FreeTypeFont] = {}

    def font(self, path: str, size: int) -> ImageFont.FreeTypeFont:
        key = (path, size)
        if key not in self._font_cache:
            self._font_cache[key] = ImageFont.truetype(path, size)
        return self._font_cache[key]

    # ---- วัดขนาดข้อความที่ wrap แล้ว ------------------------------------
    def _measure(self, text: str, font: ImageFont.FreeTypeFont,
                 max_width: float, line_spacing: float) -> TextLayout:
        lines = self.wrapper.wrap(text, font, max_width, _MEASURE)
        line_height = int(font.size * line_spacing)
        text_w = max((_MEASURE.textlength(l, font=font) for l in lines), default=0)
        text_h = line_height * max(1, len(lines))
        return TextLayout(font=font, lines=lines, font_size=font.size,
                          line_height=line_height, text_width=int(text_w),
                          text_height=int(text_h))

    def _fits(self, layout: TextLayout, box_w: float, box_h: float,
              strict: bool) -> bool:
        if layout.text_height > box_h:
            return False
        if strict:
            return layout.text_width <= box_w * 1.02
        return layout.text_width <= box_w * 1.15

    # ---- API หลัก -------------------------------------------------------
    def fit(self, text: str, box_w: float, box_h: float, font_path: str,
            min_size: int = 10, max_size: int = 200, line_spacing: float = 1.28) -> TextLayout:
        text = text.strip()
        if not text:
            f = self.font(font_path, min_size)
            return TextLayout(f, [], min_size, int(min_size * line_spacing), 0, 0)

        box_w = max(10.0, box_w)
        box_h = max(10.0, box_h)
        # เพดานขนาด: ไม่ให้ใหญ่เกินความสูงกล่องมากเกินไป
        hi = max(min_size + 1, min(max_size, int(box_h * 1.2)))

        for strict in (True, False):
            best: Optional[TextLayout] = None
            lo = min_size
            while lo <= hi:
                mid = (lo + hi) // 2
                layout = self._measure(text, self.font(font_path, mid), box_w, line_spacing)
                if self._fits(layout, box_w, box_h, strict):
                    best = layout          # ใช้ได้ → ลองใหญ่ขึ้น
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best is not None:
                return best

        # สุดท้าย: ยอมรับขนาดเล็กสุด
        return self._measure(text, self.font(font_path, min_size), box_w, line_spacing)

    # ---- โหมดกำหนดขนาดเอง (สำหรับหน้า preview/แก้ไข) ---------------------
    def fixed(self, text: str, size: int, max_width: float, font_path: str,
              line_spacing: float = 1.28) -> TextLayout:
        """
        จัดวางข้อความที่ "ขนาดฟอนต์ที่ผู้ใช้กำหนด" (ไม่ auto-fit)
        ใช้เมื่อผู้ใช้ปรับขนาดฟอนต์เองผ่านหน้า preview
        (ถ้าข้อความล้นกล่อง ผู้ใช้เป็นคนเลือกเอง — ไม่บังคับให้เล็กลง)
        """
        size = max(6, int(size))
        font = self.font(font_path, size)
        max_width = max(10.0, max_width)
        lines = self.wrapper.wrap(text.strip(), font, max_width, _MEASURE)
        line_height = int(size * line_spacing)
        text_w = max((_MEASURE.textlength(l, font=font) for l in lines), default=0)
        text_h = line_height * max(1, len(lines))
        return TextLayout(font=font, lines=lines, font_size=size,
                          line_height=line_height, text_width=int(text_w),
                          text_height=int(text_h))
