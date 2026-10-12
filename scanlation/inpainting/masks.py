# -*- coding: utf-8 -*-
"""
inpainting/masks.py
===================
เครื่องมือสร้าง "mask สำหรับ inpaint" จากผลลัพธ์ของสเตจ Detection

แนวคิดสำคัญ (professional tip):
    เราไม่ควรลบ "ทั้งบอลลูน" แต่ควอลบเฉพาะ "ตัวอักษร" ที่อยู่ข้างใน
    เพื่อเก็บเส้นขอบบอลลูนและเงา/ไล่เฉดสีของภาพเดิมไว้

    build_text_mask()  -> mask ระดับตัวอักษร (เหมาะกับ LaMa)
    build_bubble_mask()-> mask ทั้งบอลลูน (เหมาะกับ fill สีพื้น / OpenCV)
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from ..config import InpaintingConfig
from ..schemas import BBox, BlockKind, Bubble
from ..utils import geometry as geo

# ประเภทที่ควรลบข้อความออก (credit ไม่ต้องลบก็ได้ — ปรับได้)
DEFAULT_KINDS = (BlockKind.SPEECH, BlockKind.NARRATION, BlockKind.SFX)


def _local_text_threshold(gray_region: np.ndarray, delta: int = 45) -> np.ndarray:
    """
    แยก "ตัวอักษร" ออกจากพื้นหลังในบอลลูน

    วิธี: เทียบแต่ละพิกเซลกับ "สีพื้นหลังมัธยฐาน" ของบริเวณนั้น
        - ต่างกันเกิน delta → นับเป็นตัวอักษร (ไม่ว่าจะดำบนขาว หรือ ขาวบนดำ)
        - วิธีนี้เก็บ "ขอบตัวอักษรแบบ anti-alias" ได้ด้วย จึงไม่เหลือ ghosting
          (ต่างจาก Otsu ที่มักตัดเฉพาะแกนตัวอักษรเข้ม ๆ แล้วเหลือขอบเทาไว้)
    """
    if gray_region.size == 0:
        return np.zeros_like(gray_region, dtype=bool)
    # เบลอเล็กน้อยลด noise จากการสแกน
    blur = cv2.GaussianBlur(gray_region, (3, 3), 0).astype(np.int16)
    bg = float(np.median(blur))
    diff = np.abs(blur - bg)
    mask = diff > max(20, int(delta))
    # กำจัดจุดรบกวนเล็ก ๆ (noise) แต่คงเส้นตัวอักษรไว้
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((2, 2), np.uint8)).astype(bool)
    return mask


def build_text_mask(
    image: np.ndarray,
    bubbles: Iterable[Bubble],
    config: Optional[InpaintingConfig] = None,
    kinds: Sequence[BlockKind] = DEFAULT_KINDS,
) -> np.ndarray:
    """
    สร้าง mask ระดับ "ตัวอักษร" จาก Bubble ทั้งหมด
    คืนค่า mask เป็น uint8 (0/255) ขนาดเท่าภาพ
    """
    cfg = config or InpaintingConfig()
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = np.zeros((h, w), dtype=np.uint8)

    for bub in bubbles:
        if bub.kind not in kinds:
            continue
        box = bub.bbox.clip(w, h)
        x0, y0, x1, y1 = box.to_int()
        if x1 <= x0 or y1 <= y0:
            continue

        region = gray[y0:y1, x0:x1]
        local_text = _local_text_threshold(region, delta=cfg.text_delta)

        # ถ้ามี mask ของบอลลูน (จาก SAM) ให้จำกัดการลบให้อยู่ในบอลลูนเท่านั้น
        if bub.mask is not None:
            bub_region = bub.mask[y0:y1, x0:x1]
            if bub_region.shape == local_text.shape:
                local_text = local_text & bub_region

        mask[y0:y1, x0:x1] |= local_text.astype(np.uint8) * 255

    return geo.dilate_mask(mask, cfg.mask_dilate)


def build_bubble_mask(
    bubbles: Iterable[Bubble],
    height: int,
    width: int,
    config: Optional[InpaintingConfig] = None,
    kinds: Sequence[BlockKind] = DEFAULT_KINDS,
) -> np.ndarray:
    """mask ทั้งบอลลูน (ใช้ mask จาก SAM ถ้ามี; ถ้าไม่มีใช้กรอบ)"""
    cfg = config or InpaintingConfig()
    mask = np.zeros((height, width), dtype=np.uint8)
    for bub in bubbles:
        if bub.kind not in kinds:
            continue
        if bub.mask is not None and bub.mask.shape == (height, width):
            mask |= bub.mask.astype(np.uint8) * 255
        else:
            b = bub.bbox.clip(width, height).to_int()
            mask[b[1]:b[3], b[0]:b[2]] = 255
    return geo.dilate_mask(mask, cfg.mask_dilate)


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """รวมหลาย mask เข้าด้วยกันด้วย OR"""
    out: Optional[np.ndarray] = None
    for m in masks:
        if m is None:
            continue
        mm = (m > 0).astype(np.uint8)
        out = mm if out is None else (out | mm)
    if out is None:
        return np.zeros((0, 0), dtype=np.uint8)
    return out * 255
