# -*- coding: utf-8 -*-
"""
pipeline.py
===========
ScanlationPipeline — "ตัวควบคุมวงจร" (orchestrator) ที่เชื่อมทั้ง 4 สเตจเข้าด้วยกัน

    ┌─────────────────┐   ┌──────────────┐   ┌───────────────┐   ┌───────────────┐
    │ 1. Detection &  │   │ 2. Inpainting│   │ 3. Translation│   │ 4. Typesetting│
    │    Masking      │──▶│    (LaMa)    │──▶│  (Claude/4o)  │──▶│   (Pillow)    │
    │ (OCR + SAM)     │   │              │   │               │   │               │
    └─────────────────┘   └──────────────┘   └───────────────┘   └───────────────┘
         Page+Bubble          cleaned img        Bubble.th         out image

หมายเหตุการออกแบบ:
    - ทุกสเตจถูก "ฉีด" (dependency injection) เข้ามา → สลับโมเดลได้จาก config
    - ระหว่างสเตจบันทึก intermediate (JSON + ภาพ) ไว้ใน work_dir เพื่อตรวจ/ทำซ้ำ
    - เมธอด process_page() ไม่มีสถานะ → เรียกซ้อน/ขนานได้
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .config import ScanlationConfig, TypesettingConfig, TranslationConfig, InpaintingConfig, DetectionConfig
from .detection import DetectionPipeline
from .detection.ocr_detector import EasyOCRDetector
from .detection.sam_segmenter import SAMSegmenter
from .inpainting import Inpainter, LamaInpainter, build_text_mask
from .schemas import Page
from .translation import LLMTranslator, Translator, load_context
from .typesetting import Typesetter
from .utils import image_io
from .utils.logging_utils import get_logger

log = get_logger(__name__)

# นามสกุลภาพที่รองรับ
IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class ScanlationResult:
    page_index: int
    input_path: str
    output_path: Optional[str]
    page: Page


class ScanlationPipeline:
    def __init__(
        self,
        config: Optional[ScanlationConfig] = None,
        detection: Optional[DetectionPipeline] = None,
        inpainter: Optional[Inpainter] = None,
        translator: Optional[Translator] = None,
        typesetter: Optional[Typesetter] = None,
    ):
        self.cfg = config or ScanlationConfig()

        # --- สร้างคอมโพเนนต์เริ่มต้นจาก config (ถ้าไม่ได้ฉีดเข้ามา) ---
        self.detection = detection or self._default_detection()
        self.inpainter = inpainter or self._default_inpainter()
        self.translator = translator or self._default_translator()
        self.typesetter = typesetter or Typesetter(self.cfg.typesetting)

    # ------------------------------------------------------------------
    # Factory: สร้างคอมโพเนนต์จาก config
    # ------------------------------------------------------------------
    def _default_detection(self) -> DetectionPipeline:
        d = self.cfg.detection
        if d.ocr_engine == "tesseract":
            from .detection.ocr_detector import TesseractDetector
            detector = TesseractDetector(d)
        else:
            detector = EasyOCRDetector(d, gpu=d.sam_device.startswith("cuda"))

        segmenter = SAMSegmenter(d)      # เปลี่ยนเป็น FastSAMSegmenter/BoxPromptSegmenter ได้
        return DetectionPipeline(detector, segmenter, d)

    def _default_inpainter(self) -> Inpainter:
        i = self.cfg.inpainting
        if i.method == "lama":
            return LamaInpainter(i)
        if i.method == "opencv":
            from .inpainting.opencv_inpainter import OpenCVInpainter
            return OpenCVInpainter(i)
        if i.method == "median":
            from .inpainting.opencv_inpainter import MedianFillInpainter
            return MedianFillInpainter(i)
        from .inpainting.lama_inpainter import IOPaintLaMaInpainter
        return IOPaintLaMaInpainter(i, model_name=i.method)

    def _default_translator(self) -> Translator:
        return LLMTranslator(self.cfg.translation)

    # ------------------------------------------------------------------
    # ประมวลผล 1 หน้า
    # ------------------------------------------------------------------
    def process_page(self, image_path: str | Path, index: int = 0,
                     do_translate: bool = True, save: bool = True) -> ScanlationResult:
        image_path = Path(image_path)
        log.info("=== หน้า %d : %s ===", index, image_path.name)
        image = image_io.load_image_rgb(image_path)

        # (1) Detection & Masking — OCR + SAM
        page = self.detection.detect_page(image, index=index, image_path=str(image_path))
        self._dump_work(page, "1_detection")

        # (3) Translation — แปล *ก่อน* inpaint เพื่อให้ยังมีข้อความต้นฉบับให้ LLM อ่าน
        if do_translate:
            page = self.translator.translate_page(page)
            self._dump_work(page, "3_translation")

        # (2) Inpainting — ลบข้อความต้นฉบับ (ใช้ mask จาก SAM)
        mask = build_text_mask(image, page.bubbles, self.cfg.inpainting)
        cleaned = self.inpainter.inpaint(image, mask)
        image_io.save_image(cleaned, self._work_path(f"2_cleaned_p{index:04d}.png"))

        # (4) Typesetting — วาดคำแปลไทยทับ
        final = self.typesetter.render(cleaned, page)

        out_path: Optional[str] = None
        if save:
            out_path = str(self._output_path(image_path, index))
            image_io.save_image(final, out_path)
            log.info("บันทึกผลลัพธ์ -> %s", out_path)

        return ScanlationResult(page_index=index, input_path=str(image_path),
                                output_path=out_path, page=page)

    # ------------------------------------------------------------------
    # ประมวลผลหลายหน้า / ทั้งเรื่อง
    # ------------------------------------------------------------------
    def run(self, pages: Optional[Sequence[int]] = None,
            do_translate: bool = True) -> List[ScanlationResult]:
        paths = self.discover_pages()
        if pages is not None:
            wanted = set(pages)
            paths = [(i, p) for (i, p) in paths if i in wanted]
        results = []
        for index, path in paths:
            results.append(self.process_page(path, index=index, do_translate=do_translate))
        return results

    def discover_pages(self) -> List[tuple[int, Path]]:
        """หาไฟล์ภาพใน input_dir แล้วเรียงตามเลขหน้า (page-0001.webp → 1)"""
        import re

        root = Path(self.cfg.paths.input_dir)
        found: List[tuple[int, Path]] = []
        for p in sorted(root.glob("*")):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            m = re.search(r"(\d+)", p.stem)
            idx = int(m.group(1)) if m else len(found) + 1
            found.append((idx, p))
        found.sort(key=lambda t: t[0])
        return found

    # ------------------------------------------------------------------
    # helper: path + dump intermediate
    # ------------------------------------------------------------------
    def _work_path(self, name: str) -> Path:
        p = Path(self.cfg.paths.work_dir) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _output_path(self, image_path: Path, index: int) -> Path:
        out_dir = Path(self.cfg.paths.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"page-{index:04d}.png"

    def _dump_work(self, page: Page, stage: str) -> None:
        try:
            page.save_json(self._work_path(f"{stage}_p{page.index:04d}.json"))
        except Exception as exc:  # noqa: BLE001 - การ dump ไม่ควรทำให้ pipeline ล้ม
            log.warning("บันทึก intermediate ไม่สำเร็จ (%s): %s", stage, exc)
