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

    # ---- อ่านค่าที่ผู้ใช้แก้จากหน้า preview (เก็บใน Bubble.meta) -----------
    @staticmethod
    def _override_style(style: TextStyle, bubble: Bubble) -> TextStyle:
        """
        ปรับ TextStyle ตามค่าที่ผู้ใช้ตั้งไว้ในหน้า preview
        (เก็บใน bubble.meta) ได้แก่ align / color / stroke_width / line_spacing
        """
        meta = bubble.meta or {}
        align = meta.get("align")
        if align in ("left", "center", "right"):
            style.align = align
        color = meta.get("color")
        if isinstance(color, (list, tuple)) and len(color) == 3:
            style.color = (int(color[0]), int(color[1]), int(color[2]))
        if "stroke_width" in meta:
            style.stroke_width = max(0, int(meta["stroke_width"]))
        if "line_spacing" in meta:
            try:
                style.line_spacing = float(meta["line_spacing"])
            except (TypeError, ValueError):
                pass
        return style

    @staticmethod
    def _resolve_box(bubble: Bubble, w: int, h: int) -> "BBox":
        """ใช้ bbox ที่ผู้ใช้แก้ไว้ (meta['bbox']) ถ้ามี ไม่งั้นใช้ bbox เดิม"""
        from ..schemas import BBox

        meta = bubble.meta or {}
        bbox = bubble.bbox
        ov = meta.get("bbox")
        if isinstance(ov, (list, tuple)) and len(ov) == 4:
            bbox = BBox.from_list([float(v) for v in ov])
        return bbox.clip(w, h)

    # ---- เรนเดอร์บอลลูนเดียว --------------------------------------------
    def _render_bubble(self, base: Image.Image, image: np.ndarray, bubble: Bubble) -> None:
        text = bubble.translation.strip()
        if not text:
            return

        w, h = base.size
        box = self._resolve_box(bubble, w, h)
        x0, y0, x1, y1 = box.to_int()
        bw, bh = x1 - x0, y1 - y0
        if bw <= 2 or bh <= 2:
            return

        style = self._override_style(self._style_for(bubble), bubble)
        meta = bubble.meta or {}

        # ระยะขอบภายในบอลลูน (ห้ามข้อความชนขอบ)
        pad = max(4, int(style.padding_ratio * min(bw, bh)))
        ux0, uy0, ux1, uy1 = x0 + pad, y0 + pad, x1 - pad, y1 - pad
        uw, uh = max(6, ux1 - ux0), max(6, uy1 - uy0)

        # fit ขนาดฟอนต์: ถ้าผู้ใช้กำหนด font_size มา ใช้ค่านั้น (ไม่ auto-fit)
        forced_size = meta.get("font_size")
        if forced_size:
            layout = self.fitter.fixed(text, int(forced_size), uw, style.font_path,
                                       line_spacing=style.line_spacing)
        else:
            layout = self.fitter.fit(text, uw, uh, style.font_path,
                                     min_size=style.min_size, max_size=style.max_size,
                                     line_spacing=style.line_spacing)

        # SFX: คำนวณความหนาเส้นขอบจากขนาดฟอนต์ที่ fit ได้ (ถ้าผู้ใช้ไม่กำหนดเอง)
        if style.stroke_width < 0:
            style.stroke_width = max(2, int(layout.font_size * self.cfg.sfx_stroke_ratio))

        # ออฟเซ็ต (ย้ายข้อความให้ตรงกรอบ) จาก meta['offset'] = [dx, dy]
        offset = meta.get("offset")
        dx = dy = 0.0
        if isinstance(offset, (list, tuple)) and len(offset) == 2:
            dx, dy = float(offset[0]), float(offset[1])

        draw_box = (x0 + int(dx), y0 + int(dy), x1 + int(dx), y1 + int(dy))

        bg_lum = self.renderer.background_luminance(
            image, (x0, y0, x1, y1), bubble.mask
        )

        # ถ้ามีการย้ายตำแหน่ง (offset) ให้วาดแบบไม่ clip (ผู้ใช้จัดตำแหน่งเองอยู่แล้ว)
        use_clip = style.clip_to_mask and bubble.mask is not None and dx == 0 and dy == 0
        if use_clip:
            layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
            self.renderer.draw_layout(layer, layout, style, draw_box, bg_lum)
            out = self.renderer.composite_clipped(base, layer, bubble.mask, feather=1)
            base.paste(out, (0, 0))
        else:
            self.renderer.draw_layout(base, layout, style, draw_box, bg_lum)

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
