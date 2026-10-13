# -*- coding: utf-8 -*-
"""
inpainting/lama_torch_inpainter.py
==================================
ตัวลบข้อความด้วย "LaMa จริง" โดยโหลดโมเดลเป็น TorchScript ตรง ๆ

ทำไมต้องมีตัวนี้ (แยกจาก simple-lama-inpainting):
    แพ็กเกจ `simple-lama-inpainting` ตรึง `pillow<10` / `numpy<2` จึงติดตั้งไม่ได้
    บน Python 3.13+ (เช่น macOS + Python 3.14 ที่ผู้ใช้เจอ) ทำให้ต้องถอยไปใช้
    median/opencv ซึ่ง "คุณภาพใช้ไม่ได้" (ถมสีทึบ)

    โมดูลนี้โหลดไฟล์โมเดล **big-lama.pt** (TorchScript) ผ่าน `torch.jit.load`
    แล้วรัน inference เอง — ไม่พึ่ง dependency เก่าใด ๆ จึงใช้ได้กับ Python/torch
    เวอร์ชันใหม่ทั้งหมด และให้ผลลัพธ์คุณภาพเท่ากับ LaMa ต้นฉบับ

การใช้งาน:
    from scanlation.inpainting import LamaTorchInpainter
    inp = LamaTorchInpainter(cfg_inpainting)
    cleaned = inp.inpaint(image_rgb, text_mask)

ดาวน์โหลดโมเดล:
    python scripts/download_weights.py --lama      # -> weights/big-lama.pt (~205 MB)
    (หรือตัวในpaint จะดาวน์โหลดให้อัตโนมัติเมื่อยังไม่มีไฟล์)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import InpaintingConfig
from ..utils.logging_utils import get_logger
from ..utils.registry import register
from .base import Inpainter

log = get_logger(__name__)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

# แหล่งโมเดล TorchScript (ตัวเดียวกับที่ simple-lama-inpainting ใช้)
LAMA_TORCH_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/"
    "v0.1.0/big-lama.pt"
)


def _pad_to_multiple(image: np.ndarray, mask: np.ndarray, m: int = 8):
    """pad ภาพ+mask ให้ h,w หารด้วย m ลงตัว (ด้วย reflection) แล้วคืนขนาดเดิม"""
    h, w = image.shape[:2]
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph == 0 and pw == 0:
        return image, mask, (h, w)
    image_p = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mask_p = np.pad(mask, ((0, ph), (0, pw)), mode="edge")
    return image_p, mask_p, (h, w)


@register("inpainter", "lama-torch")
class LamaTorchInpainter(Inpainter):
    """LaMa (TorchScript) โหลดตรงด้วย torch.jit — ใช้ได้บน Python 3.13+"""

    def __init__(self, config: Optional[InpaintingConfig] = None, model_handle=None):
        self.cfg = config or InpaintingConfig()
        self._model = model_handle
        self._device = None
        if self._model is None:
            self._load()

    # ---- จัดการอุปกรณ์ (cuda/mps/cpu) พร้อมถอยเป็น cpu ถ้าใช้ไม่ได้ --------
    def _resolve_device(self) -> str:
        if torch is None:
            return "cpu"
        want = (self.cfg.device or "auto").lower()

        def has_cuda() -> bool:
            return bool(torch.cuda.is_available())

        def has_mps() -> bool:
            return bool(getattr(torch.backends, "mps", None)
                        and torch.backends.mps.is_available())

        if want == "auto":
            if has_cuda():
                return "cuda"
            if has_mps():          # macOS (Apple Silicon)
                return "mps"
            return "cpu"
        if want.startswith("cuda") and not has_cuda():
            # ไม่มี CUDA -> ลอง MPS (macOS) ก่อน แล้วค่อย CPU
            if has_mps():
                log.warning("ไม่พบ CUDA แต่พบ MPS — ใช้ MPS แทน")
                return "mps"
            log.warning("ไม่พบ CUDA — ใช้ CPU แทน (จะช้า)")
            return "cpu"
        if want == "mps" and not has_mps():
            log.warning("ไม่พบ MPS — ใช้ CPU แทน")
            return "cpu"
        return want

    def _checkpoint(self) -> Path:
        # ใช้ path จาก config (inpainting.lama_checkpoint) ถ้ามี ไม่งั้นค่าดีฟอลต์
        p = Path(getattr(self.cfg, "lama_checkpoint", "weights/big-lama.pt"))
        return p

    def _download(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("กำลังดาวน์โหลดโมเดล LaMa (big-lama.pt, ~205 MB) → %s ...", dest)

        def _progress(block, blk_size, total):
            if total > 0:
                pct = min(100, block * blk_size * 100 // total)
                print(f"\r  ดาวน์โหลด LaMa: {pct:3d}%", end="", flush=True)

        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(LAMA_TORCH_URL, tmp, reporthook=_progress)
        print()
        tmp.replace(dest)

    def _load(self):
        if torch is None:  # pragma: no cover
            raise RuntimeError("ต้องติดตั้ง PyTorch: pip install torch torchvision")
        ckpt = self._checkpoint()
        if not ckpt.exists():
            if getattr(self.cfg, "lama_auto_download", True):
                try:
                    self._download(ckpt)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"ดาวน์โหลดโมเดล LaMa อัตโนมัติไม่สำเร็จ: {exc}\n"
                        f"  ดาวน์โหลดเองได้ที่: {LAMA_TORCH_URL}\n"
                        f"  แล้ววางไว้ที่: {ckpt}"
                    ) from exc
            else:
                raise RuntimeError(
                    f"ไม่พบไฟล์โมเดล LaMa: {ckpt}\n"
                    f"  ดาวน์โหลดด้วย: python scripts/download_weights.py --lama\n"
                    f"  (หรือตั้ง inpainting.lama_auto_download = true)"
                )
        self._device = self._resolve_device()
        log.info("กำลังโหลด LaMa (TorchScript) device=%s จาก %s ...", self._device, ckpt)
        try:
            self._model = torch.jit.load(str(ckpt), map_location=self._device)
        except Exception as exc:  # noqa: BLE001 - MPS อาจโหลด jit ไม่ได้
            if self._device != "cpu":
                log.warning("โหลดบน %s ไม่ได้ (%s) — ลอง CPU", self._device, exc)
                self._device = "cpu"
                self._model = torch.jit.load(str(ckpt), map_location="cpu")
            else:
                raise
        self._model.eval()

    # ---- inference -------------------------------------------------------
    def _run_model(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """รัน LaMa กับภาพ+mask ขนาดเล็ก (เรียกจาก _inpaint_block)"""
        img_p, mask_p, (h, w) = _pad_to_multiple(image, mask, self.cfg.pad_to_multiple)
        img_t = torch.from_numpy(img_p.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        msk_t = torch.from_numpy(mask_p.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        img_t = img_t.to(self._device)
        msk_t = msk_t.to(self._device)
        with torch.no_grad():
            out = self._model(img_t, msk_t)
        out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return (out * 255).astype(np.uint8)[:h, :w]

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        ลบข้อความแบบ "แบ่งพื้นที่" (region-based) เพื่อความเร็วและประหยัดหน่วยความจำ:
          - หา connected component ของ mask แล้ว inpaint เฉพาะบริเวณนั้น + padding
          - ทำให้รันได้เร็วบน CPU และรองรับหน้าสูง ๆ (webtoon 800x5000) ได้
        """
        if mask.sum() == 0:
            return image.copy()

        import cv2

        mask_u8 = self._to_u8(mask)
        # เชื่อมพิกเซลตัวอักษรที่อยู่ใกล้กันให้เป็นก้อนเดียว (ลดจำนวนรอบ inference)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        merged = cv2.dilate(mask_u8, kernel, iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats((merged > 0).astype(np.uint8), 8)
        pad = max(24, int(getattr(self.cfg, "lama_region_pad", 48)))
        out = image.copy()

        for i in range(1, n):  # 0 = พื้นหลัง
            x, y, w, h, area = stats[i]
            if area <= 0:
                continue
            # ขยายกรอบ (padding) แล้วคลิปให้อยู่ในภาพ
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(image.shape[1], x + w + pad); y1 = min(image.shape[0], y + h + pad)
            crop = out[y0:y1, x0:x1]
            crop_mask = mask_u8[y0:y1, x0:x1]
            if crop_mask.sum() == 0:
                continue
            fixed = self._run_model(crop, crop_mask)
            # วางเฉพาะบริเวณที่ mask (กันการแก้พิกเซลรอบนอกที่ไม่จำเป็น)
            sel = crop_mask > 0
            out[y0:y1, x0:x1][sel] = fixed[sel]

        return out
