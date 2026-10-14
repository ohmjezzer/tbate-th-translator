#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diag_compress.py
========================
ทดสอบโมดูลบีบอัดไฟล์ภาพ (scanlation/utils/image_compressor.py) + การเชื่อมกับ
pipeline และฝั่ง Preview

    python scripts/diag_compress.py

ตรวจว่า:
    1. บีบอัด webp/jpeg ให้ไม่เกินเป้าหมาย (0.3 / 1 / 2 MB) ได้จริง
    2. หา "quality สูงสุด" ที่ยังไม่เกินเป้า (binary search)
    3. ถ้าลด quality ไม่พอ → ย่อขนาดภาพ (downscale) แทน
    4. ไฟล์เดิมที่เล็กกว่าเป้าอยู่แล้ว ไม่ถูกบีบซ้ำ (คงคุณภาพ)
    5. pipeline: บันทึกผลลัพธ์ + บีบอัดให้ไม่เกิน cfg.output.target_size_mb (ดีฟอลต์ 1MB)
    6. Preview: POST /render รับ target_mb/format แล้วได้ไฟล์ไม่เกินเป้า
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanlation.config import ScanlationConfig                       # noqa: E402
from scanlation.schemas import BBox, BlockKind, Bubble, Page, TextBlock  # noqa: E402
from scanlation.utils import image_io                               # noqa: E402
from scanlation.utils.image_compressor import (                      # noqa: E402
    compress_image_bytes,
    compress_image_file,
    human_size,
    normalize_format,
    target_bytes_from_mb,
)

PASS, FAIL = 0, 0


def check(name: str, ok: bool, extra: str = "") -> None:
    global PASS, FAIL
    ok = bool(ok)
    PASS += ok
    FAIL += (not ok)
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {extra}" if extra and not ok else ""))


def noisy_page(w: int = 1400, h: int = 2000) -> np.ndarray:
    """สร้างภาพ 'บีบอัดยาก' (ไล่สี + noise) เพื่อทดสอบขีดจำกัดการบีบอัด"""
    img = np.zeros((h, w, 3), np.uint8)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    img[..., 0] = (xx % 256); img[..., 1] = (yy % 256); img[..., 2] = ((xx + yy) % 256)
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(int) + rng.integers(-40, 40, img.shape), 0, 255).astype(np.uint8)
    return img


def main() -> int:
    work = Path("/tmp/compress_test")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    print("=== 1) บีบอัด bytes ให้ไม่เกินเป้าหมาย (webp/jpeg) ===")
    img = noisy_page()
    for mb in (0.3, 1.0, 2.0):
        target = target_bytes_from_mb(mb)
        for fmt in ("webp", "jpeg"):
            data, q, scale = compress_image_bytes(img, target, fmt=fmt)
            check(f"target {mb}MB {fmt}: {human_size(len(data))} ≤ เป้า",
                  len(data) <= target, f"ได้ {len(data)} > {target}")

    print("=== 2) เป้าเล็กมาก → ต้องย่อขนาดอัตโนมัติ (downscale) ===")
    target_small = target_bytes_from_mb(0.25)
    data, q, scale = compress_image_bytes(img, target_small, fmt="webp")
    check("ย่อขนาดลงจริง (scale < 1.0)", scale < 1.0, f"scale={scale}")
    check("ยังไม่เกินเป้าเล็ก 0.25MB", len(data) <= target_small, f"ได้ {human_size(len(data))}")

    print("=== 3) เป้าใหญ่ → ใช้ quality สูง (ไม่ลดเกินจำเป็น) ===")
    t_big = target_bytes_from_mb(3.0)
    data_big, q_big, scale_big = compress_image_bytes(img, t_big, fmt="webp")
    check("quality สูง (>= 85)", (q_big or 0) >= 85, f"q={q_big}")
    check("ไม่ย่อขนาด (scale == 1.0)", scale_big == 1.0, f"scale={scale_big}")

    print("=== 4) ไฟล์เล็กกว่าเป้าอยู่แล้ว → ไม่บีบซ้ำ ===")
    small_img = np.full((200, 200, 3), 255, np.uint8)
    sp = work / "tiny.webp"
    image_io.save_image(small_img, sp)
    before = sp.stat().st_size
    res = compress_image_file(sp, target_bytes_from_mb(1.0), fmt="webp")
    check("คงไฟล์เดิมไว้ (path เดิม)", res.path == sp, str(res.path))
    check("ขนาดไม่เปลี่ยน", res.size_bytes == before, f"{res.size_bytes} vs {before}")
    check("quality = None (ไม่ได้บีบ)", res.quality is None)

    print("=== 5) เปลี่ยนฟอร์แมต + ลบไฟล์เก่า (png -> webp) ===")
    png_path = work / "page.webp"
    image_io.save_image(img, work / "page.png")
    (work / "page.png").rename(png_path.with_suffix(".png"))
    res2 = compress_image_file(work / "page.png", target_bytes_from_mb(0.5),
                               fmt="webp", out_path=png_path, keep_original_if_smaller=False)
    check("ได้ไฟล์ .webp ใหม่", res2.path.suffix == ".webp", str(res2.path))
    check("ไฟล์ไม่เกินเป้า", res2.ok, human_size(res2.size_bytes))
    check("ลบ .png เดิมทิ้ง", not (work / "page.png").exists())

    print("=== 6) helper: normalize_format / target_bytes_from_mb ===")
    check("normalize 'jpg' -> jpeg", normalize_format("jpg") == "jpeg")
    check("normalize อะไรก็ไม่รู้ -> webp", normalize_format("xyz") == "webp")
    check("target_bytes_from_mb(1.0) = 1,000,000", target_bytes_from_mb(1.0) == 1_000_000)

    # ------------------------------------------------------------------
    print("=== 7) pipeline: บีบอัดอัตโนมัติให้ไม่เกิน 1MB (ดีฟอลต์) ===")
    proj = Path("/tmp/compress_pipeline")
    shutil.rmtree(proj, ignore_errors=True)
    (proj / "orig").mkdir(parents=True)
    image_io.save_image(noisy_page(1200, 1700), proj / "orig" / "page-0001.png")

    class MockDetection:
        def detect_page(self, image, index=0, image_path=""):
            bbox = BBox.from_list([100, 100, 500, 300])
            blk = TextBlock(uid="t0", bbox=BBox.from_list([120, 120, 480, 280]),
                            source_text="HELLO WORLD", confidence=0.9)
            return Page(index=index, image_path=image_path,
                        width=image.shape[1], height=image.shape[0],
                        bubbles=[Bubble(uid="bub-1", bbox=bbox, kind=BlockKind.SPEECH,
                                        blocks=[blk])])

    class MockTranslator:
        def translate_page(self, page):
            for b in page.bubbles:
                b.set_translation("สวัสดีชาวโลก")
            return page

    cfg = ScanlationConfig()
    cfg.paths.input_dir = str(proj / "orig")
    cfg.paths.output_dir = str(proj / "out")
    cfg.paths.work_dir = str(proj / ".scanlation")
    cfg.inpainting.method = "median"
    cfg.inpainting.mask_dilate = 0
    # ใช้ค่าดีฟอลต์ output: compress=True, target=1MB, format=webp
    check("ดีฟอลต์ target = 1.0 MB", cfg.output.target_size_mb == 1.0)
    check("ดีฟอลต์ format = webp", cfg.output.format == "webp")
    check("ดีฟอลต์ compress = True", cfg.output.compress is True)

    from scanlation.pipeline import ScanlationPipeline
    pipe = ScanlationPipeline(cfg, detection=MockDetection(), translator=MockTranslator())
    res = pipe.process_page(proj / "orig" / "page-0001.png", index=1, do_translate=True, save=True)
    out = Path(res.output_path)
    check("ผลลัพธ์เป็น .webp (ตาม format)", out.suffix == ".webp", out.name)
    check("ไฟล์ผลลัพธ์ ≤ 1MB", out.stat().st_size <= target_bytes_from_mb(1.0),
          human_size(out.stat().st_size))

    # ปิดบีบอัด -> ไฟล์ใหญ่กว่าเดิม (พิสูจน์ว่าการบีบอัดทำงานจริง)
    cfg2 = ScanlationConfig()
    cfg2.paths = cfg.paths
    cfg2.inpainting.method = "median"
    cfg2.inpainting.mask_dilate = 0
    cfg2.output.compress = False
    cfg2.output.format = "png"
    pipe2 = ScanlationPipeline(cfg2, detection=MockDetection(), translator=MockTranslator())
    res2 = pipe2.process_page(proj / "orig" / "page-0001.png", index=1, do_translate=True, save=True)
    out2 = Path(res2.output_path)
    check("ปิดบีบอัด -> ได้ .png", out2.suffix == ".png", out2.name)
    check("ไฟล์ไม่บีบอัดใหญ่กว่าไฟล์บีบอัด",
          out2.stat().st_size > out.stat().st_size,
          f"{human_size(out2.stat().st_size)} vs {human_size(out.stat().st_size)}")

    # ------------------------------------------------------------------
    print("=== 8) Preview API: POST /render รับ target_mb/format ===")
    try:
        from flask import Flask  # noqa
    except ImportError:
        print("  ⏭ ไม่มี flask — ข้ามส่วน API")
    else:
        from scanlation.web.app import create_app
        cfg3 = ScanlationConfig()
        cfg3.paths = cfg.paths
        cfg3.inpainting.method = "median"
        cfg3.output.compress = True
        cfg3.output.target_size_mb = 1.0
        cfg3.output.format = "webp"
        app = create_app(cfg3)
        c = app.test_client()

        r = c.get("/api/output-settings")
        check("GET /api/output-settings = 200", r.status_code == 200)
        check("  คืน target_mb = 1.0", r.get_json().get("target_mb") == 1.0)

        r = c.post("/api/page/1/render", json={"target_mb": 0.5, "format": "jpeg"})
        check("POST /render (target 0.5MB, jpeg) = 200", r.status_code == 200, str(r.get_json()))
        info = r.get_json().get("info", {})
        check("  บีบอัดทำงาน (compressed=True)", info.get("compressed") is True, str(info))
        check("  ได้ฟอร์แมต jpeg", info.get("format") == "jpeg", str(info))
        check("  ขนาด ≤ 0.5MB", (info.get("size_bytes") or 9e9) <= target_bytes_from_mb(0.5),
              human_size(info.get("size_bytes", 0)))

        r = c.get("/api/page/1")
        od = r.get_json().get("output", {})
        check("GET /page คืน output.size_human", bool(od.get("size_human")), str(od))
        check("  under_target = True", od.get("under_target") is True, str(od))

    print(f"\nผลรวม: ✅ ผ่าน {PASS} | ❌ ไม่ผ่าน {FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
