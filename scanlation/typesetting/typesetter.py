# -*- coding: utf-8 -*-
"""
typesetting/typesetter.py
=========================
Typesetter — ตัวควบคุมสเตจ Typesetting ให้ครบวงจร

ลำดับงานต่อ 1 บอลลูน:
    1. เลือกสไตล์ตาม "ประเภท" (speech / narration / sfx)
    2. คำนวณระยะขอบ (padding) ภายในบอลลูน
    3. fit ขนาดฟอนต์ให้พอดี (ผ่าน FontFitter)
    4. เลือกสีตัวอักษรจากความสว่างพื้นหลัง (auto contrast)
    5. วาดลงภาพ (clip ให้อยู่ใน mask ของบอลลูน)

ผลลัพธ์: ภาพที่ข้อความไทยวางทับตำแหน่งเดิมอย่างเนียน ไม่ล้นบอลลูน
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from ..config import TypesettingConfig
from ..schemas import BlockKind, Bubble, Page
from ..utils import image_io
from ..utils.logging_utils import get_logger
from .base import TextLayout, TextStyle
from .fitter import FontFitter
from .renderer import PillowRenderer
from .thai_wrapper import ThaiTextWrapper

log = get_logger(__name__)


class Typesetter:
    def __init__(self, config: Optional[TypesettingConfig] = None,
                 wrapper: Optional[ThaiTextWrapper] = None,
                 fitter: Optional[FontFitter] = None,
                 renderer: Optional[PillowRenderer] = None):
        self.cfg = config or TypesettingConfig()
        self.wrapper = wrapper or ThaiTextWrapper()
        self.fitter = fitter or FontFitter(self.wrapper)
        self.renderer = renderer or PillowRenderer(auto_contrast=self.cfg.auto_contrast)

    # ---- เลือกสไตล์ตามประเภท -------------------------------------------
    def _style_for(self, bubble: Bubble) -> TextStyle:
        if bubble.kind == BlockKind.SFX:
            return TextStyle(
                font_path=self.cfg.sfx_font,
                color=None, stroke_width=-1,     # -1 = คำนวณตามขนาดฟอนต์ทีหลัง
                align="center", padding_ratio=self.cfg.padding_ratio,
                line_spacing=self.cfg.line_spacing,
                min_size=self.cfg.min_font_size, max_size=self.cfg.max_font_size,
                clip_to_mask=bool(bubble.mask is not None),
            )
        # speech + narration ใช้ฟอนต์ปกติ/ตัวหนา แล้วแต่ตั้งค่า
        font_path = self.cfg.regular_font if bubble.kind == BlockKind.SPEECH else self.cfg.bold_font
        return TextStyle(
            font_path=font_path,
            color=None, stroke_width=0, align="center",
            padding_ratio=self.cfg.padding_ratio,
            line_spacing=self.cfg.line_spacing,
            min_size=self.cfg.min_font_size, max_size=self.cfg.max_font_size,
            clip_to_mask=bool(bubble.mask is not None),
        )

    # ---- เรนเดอร์บอลลูนเดียว --------------------------------------------
    def _render_bubble(self, base: Image.Image, image: np.ndarray, bubble: Bubble) -> None:
        text = bubble.translation.strip()
        if not text:
            return

        w, h = base.size
        box = bubble.bbox.clip(w, h)
        x0, y0, x1, y1 = box.to_int()
        bw, bh = x1 - x0, y1 - y0
        if bw <= 2 or bh <= 2:
            return

        style = self._style_for(bubble)

        # ระยะขอบภายในบอลลูน (ห้ามข้อความชนขอบ)
        pad = max(4, int(style.padding_ratio * min(bw, bh)))
        ux0, uy0, ux1, uy1 = x0 + pad, y0 + pad, x1 - pad, y1 - pad
        uw, uh = max(6, ux1 - ux0), max(6, uy1 - uy0)

        # fit ขนาดฟอนต์ให้พอดีกล่องที่หักขอบแล้ว
        layout = self.fitter.fit(text, uw, uh, style.font_path,
                                 min_size=style.min_size, max_size=style.max_size,
                                 line_spacing=style.line_spacing)

        # SFX: คำนวณความหนาเส้นขอบจากขนาดฟอนต์ที่ fit ได้
        if style.stroke_width < 0:
            style.stroke_width = max(2, int(layout.font_size * self.cfg.sfx_stroke_ratio))

        bg_lum = self.renderer.background_luminance(
            image, (x0, y0, x1, y1), bubble.mask
        )

        if style.clip_to_mask and bubble.mask is not None:
            # วาดลง layer โปร่งใส แล้ว clip ด้วย mask ของบอลลูน
            layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
            self.renderer.draw_layout(layer, layout, style, (x0, y0, x1, y1), bg_lum)
            out = self.renderer.composite_clipped(base, layer, bubble.mask, feather=1)
            base.paste(out, (0, 0))
        else:
            self.renderer.draw_layout(base, layout, style, (x0, y0, x1, y1), bg_lum)

    # ---- API หลัก -------------------------------------------------------
    def render(self, image: np.ndarray, page: Page) -> np.ndarray:
        """
        วาดข้อความไทยของทุกบอลลูนในหน้า ลงบนภาพ (ที่ inpaint แล้ว)
        คืนค่าเป็น numpy RGB
        """
        base = image_io.to_pil(image)
        for bubble in page.bubbles:
            self._render_bubble(base, image, bubble)
        return np.asarray(base.convert("RGB"), dtype=np.uint8)
