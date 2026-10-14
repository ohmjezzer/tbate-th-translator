# -*- coding: utf-8 -*-
"""
utils/image_compressor.py
=========================
โมดูลบีบอัดไฟล์ภาพผลลัพธ์ (หลัง typesetting) ให้ไม่เกินขนาดที่กำหนด

แนวคิด (Design rationale):
    - หลัง "บันทึกภาพที่แก้ไขเสร็จ" เราต้องการไฟล์ที่ "เล็กพอจะอัปโหลด/ส่งต่อได้"
      โดยยังคมชัดที่สุดเท่าที่จะทำได้ → ใช้วิธี **binary search หา quality สูงสุด**
      ที่ทำให้ไฟล์ยังไม่เกินเป้าหมาย (แทนการลด quality ลงคงที่แบบเดาสุ่ม)
    - ถ้าลด quality ถึงขั้นต่ำแล้วยังใหญ่เกิน → (เลือกได้) **ย่อขนาดภาพ** ลงทีละขั้น
      แล้วลองใหม่ (คุมด้วย max_downscale_ratio เพื่อไม่ให้ภาพเล็กเกินรับได้)
    - รองรับ WebP / JPEG (lossy) และ PNG (lossless)
    - ทำงานได้ทั้งกับ numpy array และไฟล์บนดิสก์ เพื่อให้เรียกใช้ง่ายจากทุกสเตจ

การใช้งาน:
    from scanlation.utils.image_compressor import compress_image_file

    result = compress_image_file("out/page-0001.png", target_bytes=1_000_000,
                                 fmt="webp", out_path="out/page-0001.webp")
    print(result.summary())
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from ..utils.logging_utils import get_logger

log = get_logger(__name__)

PathLike = Union[str, Path]

# 1 MB = 1,000,000 ไบต์ (ตามธรรมเนียมผู้ให้บริการอัปโหลดส่วนใหญ่)
BYTES_PER_MB = 1_000_000

_LOSSY_FORMATS = {"webp", "jpeg"}


# ---------------------------------------------------------------------------
# ตัวช่วยแสดงขนาดไฟล์
# ---------------------------------------------------------------------------
def human_size(n: int) -> str:
    """จัดรูปแบบขนาดไฟล์ให้อ่านง่าย เช่น 842KB, 1.02MB"""
    n = int(n)
    if n < 1000:
        return f"{n}B"
    if n < 1_000_000:
        return f"{n / 1000:.0f}KB"
    return f"{n / 1_000_000:.2f}MB"


def target_bytes_from_mb(target_mb: float, base: int = BYTES_PER_MB) -> int:
    """แปลงขนาดเป้าหมาย (MB) เป็นจำนวนไบต์"""
    return max(1, int(round(float(target_mb) * base)))


def normalize_format(fmt: str) -> str:
    """ทำให้ชื่อฟอร์แมตเป็นมาตรฐาน: webp | jpeg | png"""
    f = (fmt or "webp").strip().lower().lstrip(".")
    if f == "jpg":
        f = "jpeg"
    if f not in ("webp", "jpeg", "png"):
        f = "webp"
    return f


def ext_for(fmt: str) -> str:
    """นามสกุลไฟล์ของฟอร์แมต: .webp / .jpg / .png"""
    f = normalize_format(fmt)
    return ".jpg" if f == "jpeg" else f".{f}"


# ---------------------------------------------------------------------------
# ผลลัพธ์การบีบอัด
# ---------------------------------------------------------------------------
@dataclass
class CompressResult:
    path: Path                 # ไฟล์ผลลัพธ์สุดท้าย
    size_bytes: int            # ขนาดจริงหลังบีบอัด
    quality: Optional[int]     # quality สุดท้าย (None = lossless/ไม่ได้ใช้)
    scale: float               # อัตราส่วนย่อขนาด (1.0 = ไม่ย่อ)
    format: str                # ฟอร์แมตผลลัพธ์
    target_bytes: int          # เป้าหมายที่ตั้งไว้
    ok: bool                   # size <= target ไหม
    original_bytes: int = 0    # ขนาดก่อนบีบอัด

    def summary(self) -> str:
        parts = [f"{'✓' if self.ok else '⚠'} {self.path.name}: {human_size(self.size_bytes)}"]
        if self.quality is not None:
            parts.append(f"q={self.quality}")
        if self.scale < 1.0:
            parts.append(f"ย่อ {self.scale * 100:.0f}%")
        if self.original_bytes:
            pct = (1 - self.size_bytes / self.original_bytes) * 100
            parts.append(f"ลด {pct:.0f}% จาก {human_size(self.original_bytes)}")
        parts.append(f"[เป้า {human_size(self.target_bytes)}]")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# เข้ารหัสภาพเป็น bytes
# ---------------------------------------------------------------------------
def _to_pil(image) -> "object":
    """รับ numpy array หรือ PIL.Image → PIL.Image (RGB)"""
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")


def _resample():
    from PIL import Image
    return getattr(Image, "Resampling", Image).LANCZOS


def _encode(img, fmt: str, quality: int) -> bytes:
    """เข้ารหัส PIL.Image เป็น bytes"""
    buf = io.BytesIO()
    if fmt == "webp":
        img.save(buf, format="WEBP", quality=quality, method=6)
    elif fmt == "jpeg":
        # หมายเหตุ: ใช้ subsampling=0 (คุณภาพสีดีสุด) → ห้ามใช้ optimize=True คู่กัน
        # เพราะ Pillow จะล้มเมื่อเขียนลง BytesIO ("broken data stream")
        img.save(buf, format="JPEG", quality=quality, subsampling=0)
    else:  # png
        img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _best_quality_under(img, fmt: str, target: int,
                        min_quality: int, max_quality: int) -> Tuple[Optional[bytes], Optional[int]]:
    """
    หา quality สูงสุดที่ทำให้ไฟล์ไม่เกิน target (binary search)
    คืนค่า (bytes, quality) — คืน (None, None) ถ้าแม้ quality ต่ำสุดยังเกินเป้า
    """
    lo, hi = int(min_quality), int(max_quality)
    best: Optional[bytes] = None
    best_q: Optional[int] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        data = _encode(img, fmt, mid)
        if len(data) <= target:
            best, best_q = data, mid
            lo = mid + 1          # ลอง quality สูงขึ้น
        else:
            hi = mid - 1          # ลด quality ลง
    return best, best_q


def compress_image_bytes(
    image,
    target_bytes: int,
    fmt: str = "webp",
    min_quality: int = 40,
    max_quality: int = 92,
    allow_downscale: bool = True,
    max_downscale_ratio: float = 0.6,
    downscale_steps: int = 6,
) -> Tuple[bytes, Optional[int], float]:
    """
    บีบอัดภาพให้ได้ bytes ที่ไม่เกิน target_bytes (เท่าที่ทำได้)

    คืนค่า: (bytes, quality, scale)
        - ถ้าทำไม่ถึงเป้าแม้ลด quality+ย่อขนาดต่ำสุดแล้ว จะคืนค่าที่เล็กที่สุดที่ทำได้
    """
    img = _to_pil(image)
    fmt = normalize_format(fmt)
    target = max(1, int(target_bytes))

    # ---- ขั้น 1: lossy → binary search คุณภาพ;  lossless (png) → ลองตรง ๆ ----
    best_scale = 1.0
    if fmt in _LOSSY_FORMATS:
        top = _encode(img, fmt, int(max_quality))
        if len(top) <= target:
            return top, int(max_quality), 1.0
        best, best_q = _best_quality_under(img, fmt, target, min_quality, max_quality)
        if best is not None:
            return best, best_q, 1.0
        # ต่ำสุดยังเกิน → เก็บตัวเล็กสุดไว้เป็นทางเลือกก่อนย่อ
        best = _encode(img, fmt, int(min_quality))
        best_q = int(min_quality)
    else:
        data = _encode(img, fmt, 100)
        if len(data) <= target:
            return data, None, 1.0
        best, best_q = data, None

    # ---- ขั้น 2: ย่อขนาดภาพทีละขั้น (ค่อย ๆ ลด) ----
    if allow_downscale:
        ratio = 1.0
        step = (1.0 - float(max_downscale_ratio)) / max(1, downscale_steps)
        for _ in range(max(1, downscale_steps)):
            ratio -= step
            if ratio < float(max_downscale_ratio) - 1e-9:
                break
            w = max(1, int(round(img.width * ratio)))
            h = max(1, int(round(img.height * ratio)))
            small = img.resize((w, h), _resample())

            if fmt in _LOSSY_FORMATS:
                data, q = _best_quality_under(small, fmt, target, min_quality, max_quality)
                if data is not None:
                    return data, q, ratio
                data, q = _encode(small, fmt, int(min_quality)), int(min_quality)
            else:
                data, q = _encode(small, fmt, 100), None

            # เก็บตัวเลือกที่เล็กที่สุดไว้ (รวม scale ที่สอดคล้องกัน)
            if best is None or len(data) < len(best):
                best, best_q, best_scale = data, q, ratio

    return best, best_q, best_scale


# ---------------------------------------------------------------------------
# API หลัก: บีบอัดไฟล์บนดิสก์
# ---------------------------------------------------------------------------
def compress_image_file(
    src: PathLike,
    target_bytes: int,
    fmt: str = "webp",
    out_path: Optional[PathLike] = None,
    min_quality: int = 40,
    max_quality: int = 92,
    allow_downscale: bool = True,
    max_downscale_ratio: float = 0.6,
    keep_original_if_smaller: bool = True,
) -> CompressResult:
    """
    บีบอัดไฟล์ภาพจากดิสก์ให้ไม่เกิน target_bytes

    - ถ้าไฟล์เดิมเล็กกว่าเป้าอยู่แล้ว + นามสกุลตรงกับ fmt + keep_original_if_smaller
      → คงไฟล์เดิมไว้ (ไม่บีบอัดซ้ำให้เสียคุณภาพ)
    - out_path: ปลายทาง (ถ้าไม่ระบุ = src แต่เปลี่ยนนามสกุลตาม fmt)
    - ถ้าเขียนไปคนละไฟล์กับ src → ลบ src เดิมให้อัตโนมัติ
    """
    src = Path(src)
    fmt = normalize_format(fmt)
    original_bytes = src.stat().st_size

    if out_path is None:
        out_path = src.with_suffix(ext_for(fmt))
    out_path = Path(out_path)

    if (keep_original_if_smaller and original_bytes <= target_bytes
            and src.suffix.lower() == ext_for(fmt)):
        return CompressResult(path=src, size_bytes=original_bytes, quality=None,
                              scale=1.0, format=fmt, target_bytes=target_bytes,
                              ok=True, original_bytes=original_bytes)

    from ..utils import image_io

    image = image_io.load_image_rgb(src)
    data, quality, scale = compress_image_bytes(
        image, target_bytes, fmt=fmt, min_quality=min_quality, max_quality=max_quality,
        allow_downscale=allow_downscale, max_downscale_ratio=max_downscale_ratio)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)

    if out_path.resolve() != src.resolve() and src.exists():
        try:
            src.unlink()
        except OSError:
            pass

    res = CompressResult(path=out_path, size_bytes=len(data), quality=quality,
                         scale=scale, format=fmt, target_bytes=target_bytes,
                         ok=len(data) <= target_bytes, original_bytes=original_bytes)
    log.info("บีบอัด: %s", res.summary())
    return res


def compress_array_to_file(
    image,
    out_path: PathLike,
    target_bytes: int,
    fmt: str = "webp",
    **kwargs,
) -> CompressResult:
    """บีบอัด numpy array แล้วเขียนลงไฟล์โดยตรง (ใช้ใน pipeline)"""
    out_path = Path(out_path)
    fmt = normalize_format(fmt)
    data, quality, scale = compress_image_bytes(image, target_bytes, fmt=fmt, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return CompressResult(path=out_path, size_bytes=len(data), quality=quality,
                          scale=scale, format=fmt, target_bytes=target_bytes,
                          ok=len(data) <= int(target_bytes))
