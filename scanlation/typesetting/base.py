# -*- coding: utf-8 -*-
"""
typesetting/base.py
===================
โครงสร้างข้อมูลของสเตจ Typesetting

    TextStyle  : สไตล์การวาด (ฟอนต์ สี ระยะขอบ เส้นขอบ ฯลฯ)
    TextLayout : ผลลัพธ์การจัดวาง (ฟอนต์ที่ fit แล้ว, บรรทัด, ขนาดกล่อง)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import ImageFont


@dataclass
class TextStyle:
    """พารามิเตอร์การวาดข้อความ 1 บล็อก"""

    font_path: str
    color: Optional[Tuple[int, int, int]] = None   # None = auto contrast
    stroke_width: int = 0                          # ความหนาเส้นขอบ (0 = ไม่มี)
    stroke_fill: Optional[Tuple[int, int, int]] = None
    align: str = "center"                           # center | left | right
    padding_ratio: float = 0.14                     # ระยะขอบในบอลลูน
    line_spacing: float = 1.28
    min_size: int = 10
    max_size: int = 200
    uppercase: bool = False
    clip_to_mask: bool = True                       # ตัดข้อความไม่ให้ล้นบอลลูน


@dataclass
class TextLayout:
    """ผลลัพธ์จากการ fit ข้อความให้พอดีกล่อง"""

    font: ImageFont.FreeTypeFont
    lines: List[str]
    font_size: int
    line_height: int
    text_width: int
    text_height: int
    box: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0))  # x0,y0,x1,y1
