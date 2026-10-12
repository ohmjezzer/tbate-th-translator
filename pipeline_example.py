# -*- coding: utf-8 -*-
"""
pipeline_example.py
===================
ตัวอย่างการใช้งานจริง: เชื่อมทั้ง 5 ขั้นตอนเข้าด้วยกัน

    Download & Ingest (ดึงรูปจากเว็บ → บังคับ WebP → orig/)
        → Detection & Masking (OCR + SAM)
        → Inpainting (LaMa)
        → Translation (Claude 3.5 Sonnet / GPT-4o)
        → Typesetting (Pillow + ฟอนต์ไทย)

รันแบบเต็มระบบ:
    export ANTHROPIC_API_KEY=sk-ant-...        # หรือ OPENAI_API_KEY

    # 0) ดาวน์โหลด chapter จากเว็บ → บังคับ WebP → orig/ แล้วแปลต่อทันที
    python pipeline_example.py --chapter-url https://vortexscans.org/series/the-beginning-after-the-end/chapter-255
    python pipeline_example.py --chapter-url <url> --limit 3      # ทดลอง 3 หน้าแรก

    # 1) ลองกับภาพจำลอง (ไม่ต้องมี API key / โมเดล)
    python pipeline_example.py --demo

    # 2) ประมวลผลโฟลเดอร์ orig/ ที่มีอยู่แล้ว -> out/
    python pipeline_example.py
    python pipeline_example.py 4 23 26

อ่านโค้ดจากบนลงล่าง: แต่ละคอมเมนต์อธิบายว่า "ทำไม" ไม่ใช่แค่ "ทำอะไร"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from scanlation.config import (
    DetectionConfig,
    DownloadConfig,
    InpaintingConfig,
    ScanlationConfig,
    TranslationConfig,
    TypesettingConfig,
)
from scanlation.downloader import ChapterDownloader
from scanlation.pipeline import ScanlationPipeline


# ===========================================================================
# โหมด 0: download — ดึงรูปจากเว็บ → บังคับแปลง WebP → วางใน orig/
# ===========================================================================
def run_download(chapter_url: str, limit: int, out_dir: str | None,
                 translate: bool, pages: list[int] | None) -> None:
    """
    สาธิตสเตจ Download & Ingest:
        1) ChapterDownloader เลือก source (vortexscans/generic-*) จาก config
        2) source.list_images(url) -> รายการ ImageRef (เรียงตามหน้า)
        3) ดาวน์โหลดแบบขนาน -> ImageConverter บังคับเป็น WebP -> page-XXXX.webp
    แล้ว (ถ้า translate=True) ส่งต่อให้ pipeline แปลทันที
    """
    cfg = ScanlationConfig()
    cfg.download.limit = limit
    if out_dir:
        cfg.download.out_dir = out_dir

    # --- ใช้ ChapterDownloader ตรง ๆ ---
    downloader = ChapterDownloader(cfg.download)
    report = downloader.download_chapter(chapter_url)
    print(report.summary())
    for f in report.files[:5]:
        print("   ", f)
    if report.failed:
        print(f"   (มี {report.failed} หน้าที่ล้มเหลว)")

    if not translate:
        print("\nดาวน์โหลดเสร็จ — รันต่อด้วย:  python -m scanlation.cli run")
        return

    # --- หรือจะใช้ pipeline ทางลัด (download + run) ก็ได้ ---
    if not cfg.api_key():
        print("\n(ข้ามการแปล: ไม่พบ API key — ตั้ง ANTHROPIC_API_KEY / OPENAI_API_KEY)")
        return
    pipeline = ScanlationPipeline(cfg)
    results = pipeline.run(pages=pages)
    for r in results:
        print(f"หน้า {r.page_index:>3} → {r.output_path}")



# ===========================================================================
# โหมดที่ 1: full pipeline — ใช้คอมโพเนนต์จริงทั้งหมด (SAM + LaMa + LLM)
# ===========================================================================
def run_full(config_path: str | None, pages: list[int] | None) -> None:
    cfg = ScanlationConfig.load(config_path) if config_path else ScanlationConfig()

    # ตรวจว่ามี API key ของ LLM หรือยัง
    if not cfg.api_key():
        raise SystemExit(
            "ไม่พบ API key: กรุณาตั้งค่า ANTHROPIC_API_KEY หรือ OPENAI_API_KEY "
            "ให้ตรงกับ cfg.translation.backend"
        )

    # สร้าง pipeline (คอมโพเนนต์ถูกสร้างจาก config อัตโนมัติ)
    pipeline = ScanlationPipeline(cfg)

    # ประมวลผลเฉพาะหน้าที่ต้องการ (ถ้าไม่ระบุ = ทุกหน้าใน input_dir)
    results = pipeline.run(pages=pages)
    for r in results:
        print(f"หน้า {r.page_index:>3} → {r.output_path}  "
              f"({len(r.page.bubbles)} บอลลูน)")


# ===========================================================================
# โหมดที่ 2: demo — ใช้เฉพาะคอมโพเนนต์ที่ไม่ต้องพึ่ง GPU/API (สำหรับทดสอบโครงสร้าง)
# ===========================================================================
def run_demo() -> None:
    """
    สาธิต "การฉีด dependency" (DI):
        - สลับ SAM → BoxPromptSegmenter (ไม่ต้องมี checkpoint / GPU)
        - สลับ LaMa  → MedianFillInpainter (ไม่ต้องมี torch)
        - สลับ LLM   → DummyTranslator (ไม่ต้องมี API key)
    โดย pipeline ไม่ต้องแก้อะไรเลย — นี่คือหัวใจของสถาปัตยกรรมที่ขยายผลได้
    """
    from PIL import Image, ImageDraw, ImageFont

    from scanlation.detection.sam_segmenter import BoxPromptSegmenter
    from scanlation.detection import DetectionPipeline
    from scanlation.detection.ocr_detector import EasyOCRDetector  # จะถูกข้ามใน demo
    from scanlation.inpainting import build_text_mask, MedianFillInpainter
    from scanlation.translation.base import Translator
    from scanlation.schemas import BBox, BlockKind, TextBlock, Bubble, Page
    from scanlation.typesetting import Typesetter
    from scanlation.utils import image_io

    # ---- สร้างภาพจำลอง (บอลลูนขาว + ตัวอักษรอังกฤษ) ----
    cfg = ScanlationConfig()
    H, W = 900, 760
    canvas = np.zeros((H, W, 3), np.uint8)
    pil = Image.fromarray(canvas)
    d = ImageDraw.Draw(pil)
    d.ellipse([60, 60, 700, 400], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    f = ImageFont.truetype(cfg.typesetting.regular_font, 34)
    d.text((170, 180), "I WILL PROTECT THEM.", fill=(0, 0, 0), font=f)
    d.text((170, 240), "NO MATTER WHAT.", fill=(0, 0, 0), font=f)
    image = np.asarray(pil, np.uint8)

    # ---- จำลองผล OCR: TextBlock ที่อยู่ในบอลลูน ----
    text_blocks = [
        TextBlock(uid="t0", bbox=BBox(170, 180, 560, 215), source_text="I WILL PROTECT THEM.", confidence=0.95),
        TextBlock(uid="t1", bbox=BBox(170, 240, 500, 275), source_text="NO MATTER WHAT.", confidence=0.95),
    ]

    # ---- สเตจ 1: Detection & Masking (demo ใช้ box segmenter) ----
    seg = BoxPromptSegmenter(cfg.detection)
    bubbles = seg.segment(image, text_blocks)
    page = Page(index=1, image_path="demo", width=W, height=H, bubbles=bubbles)

    # ---- สเตจ 3: Translation (demo ใช้พจนานุกรมคงที่) ----
    class DummyTranslator(Translator):
        """แทน LLM เพื่อทดสอบโดยไม่ต้องมี API key"""
        FAKE = {
            "bub0": "ข้าจะปกป้องพวกเขาเอง\nไม่ว่าจะเกิดอะไรขึ้นก็ตาม",
        }

        def translate_page(self, page, context=None):
            for b in page.bubbles:
                b.set_translation(self.FAKE.get(b.uid, "คำแปลตัวอย่าง"))
            return page

    page = DummyTranslator().translate_page(page)

    # ---- สเตจ 2: Inpainting (demo ใช้ median fill) ----
    mask = build_text_mask(image, page.bubbles, cfg.inpainting)
    cleaned = MedianFillInpainter(cfg.inpainting).inpaint(image, mask)

    # ---- สเตจ 4: Typesetting ----
    final = Typesetter(cfg.typesetting).render(cleaned, page)

    # ---- บันทึกผล (ก่อน/หลัง เทียบกัน) ----
    Path("out").mkdir(exist_ok=True)
    image_io.save_image(
        np.concatenate([image, np.full((H, 12, 3), 255, np.uint8), final], axis=1),
        "out/demo_before_after.png",
    )
    print("เสร็จ! ดูผลได้ที่ out/demo_before_after.png")
    print("บอลลูนที่ตรวจพบ:", [(b.uid, b.kind.value) for b in page.bubbles])


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="ตัวอย่างการใช้งาน Scanlation Pipeline")
    ap.add_argument("--config", help="path ไปยัง config.json")
    ap.add_argument("--demo", action="store_true", help="รันเดโมโดยไม่ต้องใช้ GPU/API key")
    ap.add_argument("--chapter-url", help="URL ของ chapter ที่จะดาวน์โหลด+แปล (โหมด download)")
    ap.add_argument("--limit", type=int, default=0, help="จำกัดจำนวนหน้าที่ดาวน์โหลด (0=ทั้งหมด)")
    ap.add_argument("--out", help="โฟลเดอร์ปลายทางของรูปที่ดาวน์โหลด (ดีฟอลต์ orig)")
    ap.add_argument("--download-only", action="store_true", help="ดาวน์โหลดอย่างเดียว ไม่แปลต่อ")
    ap.add_argument("pages", nargs="*", help="เลขหน้าที่ต้องการ (โหมดเต็มระบบ)")
    args = ap.parse_args()

    if args.chapter_url:
        run_download(args.chapter_url, args.limit, args.out,
                     translate=not args.download_only, pages=[int(x) for x in args.pages] or None)
    elif args.demo:
        run_demo()
    else:
        pages = [int(x) for x in args.pages] if args.pages else None
        run_full(args.config, pages)


if __name__ == "__main__":
    main()
