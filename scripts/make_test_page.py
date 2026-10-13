#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/make_test_page.py
=========================
สร้าง "หน้าทดสอบสังเคราะห์" สำหรับตรวจว่า pipeline เนียนไหม โดยไม่ต้องพึ่งภาพจริง
(มีบอลลูนคำพูด + กล่องบรรยาย + SFX บนพื้นหลังไล่สี ซึ่งเป็นเคสที่พบบ่อยในเว็บตูน)

    python scripts/make_test_page.py                 # -> testdata/page-test.png
    python scripts/make_test_page.py out.png --seed 1

จากนั้นรัน:
    python scripts/diagnose.py testdata/page-test.png --segmenter box --inpaint median --no-translate
    python scripts/diag_isolate_detection.py testdata/page-test.png --inpaint median
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent


def build(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    H, W = 1400, 800

    # พื้นหลังไล่สีเข้ม + noise (จำลองภาพจริงที่มี texture)
    img = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        img[y, :] = [max(10, 40 - y // 40), max(10, 55 - y // 40), max(20, 120 - y // 25)]
    img = np.clip(img.astype(np.int16) + rng.integers(-8, 8, img.shape), 0, 255).astype(np.uint8)

    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    f = ImageFont.truetype(str(ROOT / "fonts/Sarabun-Regular.ttf"), 30)
    fb = ImageFont.truetype(str(ROOT / "fonts/Sarabun-Bold.ttf"), 40)

    # (1) บอลลูนคำพูด 2 บรรทัด
    d.ellipse([60, 60, 620, 340], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    d.text((150, 150), "I WILL PROTECT", fill=(0, 0, 0), font=f)
    d.text((150, 200), "EVERYONE I LOVE!", fill=(0, 0, 0), font=f)

    # (2) กล่องบรรยาย 2 บรรทัด
    d.rounded_rectangle([60, 440, 660, 700], radius=6, fill=(252, 252, 250),
                        outline=(0, 0, 0), width=2)
    d.text((110, 510), "But the twin horns were", fill=(0, 0, 0), font=f)
    d.text((110, 560), "nowhere to be found.", fill=(0, 0, 0), font=f)

    # (3) SFX ตัวอักษรใหญ่บนพื้นหลัง (ไม่มีบอลลูน) — อยู่ "แนวเดียวกับ" คำพูดในบอลลูน
    #     เพื่อทดสอบว่า detector ยังแยกสองก้อนนี้ออกจากกันได้หรือไม่
    d.text((680, 200), "CRACK!", fill=(80, 220, 255), font=fb)
    return pil


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="testdata/page-test.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build(args.seed).save(out)
    print(f"✓ สร้างหน้าทดสอบ: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
