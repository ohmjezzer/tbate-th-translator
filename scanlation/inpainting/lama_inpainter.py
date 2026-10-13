# -*- coding: utf-8 -*-
"""
inpainting/lama_inpainter.py
============================
ตัวลบข้อความด้วยโมเดล LaMa (Large Mask Inpainting) — ให้พื้นหลังเนียนที่สุด

มี 2 ทางเชื่อมต่อ:
    1) LamaInpainter        ใช้แพ็กเกจ `simple-lama-inpainting` (แนะนำ, API ง่าย)
       pip install simple-lama-inpainting

    2) IOPaintLaMaInpainter ใช้ `iopaint` (ชื่อเดิม lama-cleaner) ซึ่งรองรับ
       โมเดลอื่นเพิ่มได้ เช่น MAT / MI-GAN / SD (ขยายผลในอนาคต)
       pip install iopaint

เหตุผลที่ต้อง pad ขนาดภาพ:
    สถาปัตยกรรม LaMa (FFC) ต้องการ h,w หารด้วย 8 (หรือ 32) ลงตัว
    เราจึง pad ด้วย reflection แล้ว crop กลับ — ตัว wrapper จัดการให้อัตโนมัติ
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import InpaintingConfig
from ..utils import image_io
from ..utils.logging_utils import get_logger
from ..utils.registry import register
from .base import Inpainter

log = get_logger(__name__)

try:  # optional import — ผู้ใช้ที่ยังไม่ติดตั้งต้อง import โมดูลได้
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def _pad_to_multiple(image: np.ndarray, mask: np.ndarray, m: int = 8):
    """pad ภาพ+mask ให้ h,w หารด้วย m ลงตัว (ด้วย reflection) แล้วคืนค่าขนาดเดิม"""
    h, w = image.shape[:2]
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph == 0 and pw == 0:
        return image, mask, (h, w)
    image_p = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mask_p = np.pad(mask, ((0, ph), (0, pw)), mode="edge")
    return image_p, mask_p, (h, w)


@register("inpainter", "lama")
class LamaInpainter(Inpainter):
    """
    Wrapper ของ LaMa ผ่านไลบรารี simple-lama-inpainting

    ตัวอย่าง:
        inpainter = LamaInpainter(InpaintingConfig(device="cuda"))
        cleaned = inpainter.inpaint(image_rgb, text_mask)
    """

    def __init__(self, config: Optional[InpaintingConfig] = None, model_handle=None):
        self.cfg = config or InpaintingConfig()
        self._model = model_handle
        if self._model is None:
            self._load()

    def _load(self):
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ยังไม่ได้ติดตั้ง LaMa (simple-lama-inpainting)\n"
                "  ⚠️ ข้อจำกัด: แพ็กเกจนี้กำหนด pillow<10 / numpy<2 จึงติดตั้งไม่ได้บน Python 3.13+\n"
                "  เลือกทางใดทางหนึ่ง:\n"
                "    1) ใช้ inpaint สำรองที่ติดตั้งได้ทันที (ไม่ต้องมี LaMa):\n"
                "         สร้าง .env แล้วใส่  SCANLATION_INPAINT_METHOD=median\n"
                "         (หรือ opencv สำหรับคุณภาพกลาง ๆ)\n"
                "    2) ติดตั้ง LaMa บน Python 3.11/3.12:\n"
                "         python3.11 -m venv .venv && pip install -r requirements-ml.txt\n"
                "    3) ใช้ IOPaint (รองรับ Python/numpy ใหม่กว่า):\n"
                "         pip install iopaint    แล้วตั้ง  SCANLATION_INPAINT_METHOD=iopaint"
            ) from exc
        device = self.cfg.device
        if torch is not None and device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("ไม่พบ CUDA — เปลี่ยนไปใช้ CPU")
            device = "cpu"
        log.info("กำลังโหลด LaMa (device=%s)...", device)
        self._model = SimpleLama(device=device)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.sum() == 0:
            return image.copy()

        mask_u8 = self._to_u8(mask)
        # pad ให้ขนาดหารด้วย 8 ลงตัวก่อนป้อนโมเดล
        img_p, mask_p, (h, w) = _pad_to_multiple(image, mask_u8, self.cfg.pad_to_multiple)

        result = self._model(image_io.to_pil(img_p), image_io.to_pil(mask_p))
        out = np.asarray(result, dtype=np.uint8)
        if out.shape[0] != h or out.shape[1] != w:
            out = out[:h, :w]
        return out


@register("inpainter", "iopaint")
class IOPaintLaMaInpainter(Inpainter):
    """
    เชื่อมต่อผ่าน IOPaint (lama-cleaner) — รองรับหลายโมเดลในตัวเดียว
    ตั้งชื่อโมเดลผ่านพารามิเตอร์ model_name เช่น "lama", "mat", "migan", "zits"

    เหมาะเมื่ออยากลองโมเดล inpainting ใหม่ ๆ โดยไม่ต้องเขียน adapter ใหม่
    """

    def __init__(self, config: Optional[InpaintingConfig] = None,
                 model_name: str = "lama", model_handle=None):
        self.cfg = config or InpaintingConfig()
        self.model_name = model_name
        self._manager = model_handle
        if self._manager is None:
            self._load()

    def _load(self):
        try:
            from iopaint.model_manager import ModelManager
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("ต้องติดตั้ง IOPaint: pip install iopaint") from exc
        log.info("กำลังโหลด IOPaint ModelManager (name=%s, device=%s)...",
                 self.model_name, self.cfg.device)
        self._manager = ModelManager(name=self.model_name, device=self.cfg.device)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.sum() == 0:
            return image.copy()
        from iopaint.schema import InpaintRequest

        mask_u8 = self._to_u8(mask)
        img_p, mask_p, (h, w) = _pad_to_multiple(image, mask_u8, self.cfg.pad_to_multiple)
        # IOPaint: รับ numpy RGB + mask uint8 → คืน numpy RGB
        out = self._manager(img_p, mask_p, InpaintRequest())
        out = np.asarray(out, dtype=np.uint8)
        return out[:h, :w]
