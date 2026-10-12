# -*- coding: utf-8 -*-
"""
downloader/converter.py
=======================
ตัวแปลงภาพ "บังคับเป็น WebP" (Force-WebP conversion)

เหตุผลที่ต้องบังคับ:
    เว็บต้นทางอาจให้รูปมาเป็น jpg / png / avif / webp ปนกัน การทำให้เป็น
    WebP ทุกไฟล์ (a) ทำให้สเตจ downstream ทำงานสม่ำเสมอ (b) ประหยัดพื้นที่
    (c) คงคุณภาพด้วยการควบคุม quality/method

หมายเหตุ: WebP ของ Pillow รองรับ RGB เป็นหลัก -> แปลงโหมด RGBA/P/CMYK ให้เป็น RGB
ก่อนบันทึก (ภาพการ์ตูนส่วนใหญ่ไม่มีโปร่งใสอยู่แล้ว)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import DownloadConfig
from ..utils.logging_utils import get_logger

log = get_logger(__name__)


class ImageConverter:
    def __init__(self, config: DownloadConfig | None = None):
        self.cfg = config or DownloadConfig()

    def to_webp_bytes(self, raw: bytes) -> bytes:
        """แปลงข้อมูลภาพ (ไบต์) ใด ๆ → ข้อมูล WebP (ไบต์)"""
        with Image.open(io.BytesIO(raw)) as im:
            im = self._prepare(im)
            buf = io.BytesIO()
            im.save(buf, format="WEBP", quality=self.cfg.webp_quality,
                    method=self.cfg.webp_method)
            return buf.getvalue()

    def save_webp(self, raw: bytes | np.ndarray, dest: str | Path) -> int:
        """
        แปลงและบันทึกเป็นไฟล์ WebP ที่ path ปลายทาง
        คืนค่า: ขนาดไฟล์ (ไบต์)
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(raw, (bytes, bytearray)):
            data = self.to_webp_bytes(bytes(raw))
            dest.write_bytes(data)
        else:  # numpy array (RGB)
            im = self._prepare(Image.fromarray(np.asarray(raw, dtype=np.uint8), "RGB"))
            im.save(dest, format="WEBP", quality=self.cfg.webp_quality,
                    method=self.cfg.webp_method)
        return dest.stat().st_size

    # ---- ปรับโหมด/ขนาดก่อนบันทึก ---------------------------------------
    def _prepare(self, im: Image.Image) -> Image.Image:
        # 1) ถ้าภาพมี alpha ให้ flatten ลงพื้นขาว (WebP RGB ไม่มี alpha)
        if im.mode in ("RGBA", "LA", "P"):
            if im.mode == "P":
                im = im.convert("RGBA")
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
        if im.mode != "RGB":
            im = im.convert("RGB")

        # 2) ย่อขนาดถ้ากำหนด max_width
        if self.cfg.max_width and im.width > self.cfg.max_width:
            ratio = self.cfg.max_width / float(im.width)
            new_h = int(round(im.height * ratio))
            im = im.resize((self.cfg.max_width, new_h), Image.LANCZOS)
        return im
