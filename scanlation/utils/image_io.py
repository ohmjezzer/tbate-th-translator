# -*- coding: utf-8 -*-
"""
image_io.py
===========
ฟังก์ชันอ่าน/เขียนภาพแบบรวมศูนย์ เพื่อให้ทั้งโปรเจกต์ใช้รูปแบบเดียวกันเสมอ
    - ภายในระบบใช้ numpy array แบบ RGB uint8 (H x W x 3) เป็นตัวแทนภาพ
    - ภายนอก (Pillow) แปลงผ่าน helper เหล่านี้เท่านั้น
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]


def load_pil(path: PathLike) -> Image.Image:
    """เปิดภาพด้วย Pillow (คืนค่าเป็น RGB เสมอ)"""
    return Image.open(path).convert("RGB")


def load_image_rgb(path: PathLike) -> np.ndarray:
    """โหลดภาพเป็น numpy array RGB uint8"""
    return np.asarray(load_pil(path), dtype=np.uint8)


def to_pil(image: np.ndarray) -> Image.Image:
    """แปลง numpy RGB array -> PIL.Image"""
    return Image.fromarray(image.astype(np.uint8), mode="RGB")


def save_image(image: np.ndarray, path: PathLike, quality: int = 95) -> None:
    """
    บันทึกภาพ โดยเลือกฟอร์แมตจากนามสกุลไฟล์
        - .png / .webp : บันทึกแบบ lossless (หรือคุณภาพสูง)
        - .jpg / .jpeg : บันทึกด้วย quality ที่กำหนด
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = to_pil(image)
    if p.suffix.lower() in (".jpg", ".jpeg"):
        img.save(p, quality=quality, subsampling=0)
    elif p.suffix.lower() == ".webp":
        img.save(p, quality=quality, method=6)
    else:
        img.save(p)


def image_size(image: np.ndarray) -> Tuple[int, int]:
    """คืนค่า (width, height)"""
    h, w = image.shape[:2]
    return w, h
