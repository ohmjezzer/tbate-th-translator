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
from .downloader import ChapterDownloader, DownloadReport
from .inpainting import Inpainter, LamaInpainter, build_text_mask
from .schemas import Page
from .translation import LLMTranslator, Translator, load_context
from .typesetting import Typesetter
from .utils import image_io
from .utils.image_compressor import (
    compress_array_to_file,
    compress_image_file,
    ext_for,
    human_size,
    normalize_format,
    target_bytes_from_mb,
)
from .utils.logging_utils import get_logger
from .monitoring import MetricsCollector

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
        downloader: Optional[ChapterDownloader] = None,
    ):
        self.cfg = config or ScanlationConfig()

        # --- คอมโพเนนต์ที่ฉีดเข้ามา (ถ้าไม่ส่ง จะสร้างแบบ lazy ตอนเรียกใช้) ---
        # การสร้างแบบ lazy สำคัญมาก: คำสั่ง "download" ไม่ควรถูกบังคับให้โหลด
        # โมเดลหนัก (SAM/LaMa) หรือเรียก LLM เลย
        self._detection = detection
        self._inpainter = inpainter
        self._translator = translator
        self._typesetter = typesetter
        self._downloader = downloader

    # ------------------------------------------------------------------
    # Lazy accessors: สร้างคอมโพเนนต์เมื่อถูกเรียกครั้งแรก แล้ว cache ไว้
    # ------------------------------------------------------------------
    @property
    def downloader(self) -> ChapterDownloader:
        if self._downloader is None:
            self._downloader = ChapterDownloader(self.cfg.download)
        return self._downloader

    @property
    def detection(self) -> DetectionPipeline:
        if self._detection is None:
            self._detection = self._default_detection()
        return self._detection

    @property
    def inpainter(self) -> Inpainter:
        if self._inpainter is None:
            self._inpainter = self._default_inpainter()
        return self._inpainter

    @property
    def translator(self) -> Translator:
        if self._translator is None:
            self._translator = self._default_translator()
        return self._translator

    @property
    def typesetter(self) -> Typesetter:
        if self._typesetter is None:
            self._typesetter = Typesetter(self.cfg.typesetting)
        return self._typesetter

    # ------------------------------------------------------------------
    # Factory: สร้างคอมโพเนนต์จาก config
    # ------------------------------------------------------------------
    @staticmethod
    def resolve_segmenter_name(name: str, allow_fallback: bool = True,
                               detection_cfg: Optional[DetectionConfig] = None) -> str:
        """
        แปลงชื่อ segmenter จาก config ให้เป็นตัวที่ "รันได้จริง" บนเครื่องนี้

            "auto" -> เลือกตัวที่ดีที่สุดที่พร้อมใช้ (sam -> fastsam -> box)
            อื่น ๆ  -> ตรวจว่าติดตั้ง dependency และมีไฟล์โมเดลครบไหม ถ้าไม่ และ
                       auto_fallback=True จะถอยไปใช้ตัวที่รันได้แทน (พร้อมเตือน)

        detection_cfg: ส่ง DetectionConfig เข้ามาเพื่อตรวจ "ไฟล์ checkpoint ของ SAM"
        (ถ้าไม่ส่ง จะตรวจแค่ว่าติดตั้งโมดูล segment-anything หรือไม่)
        """
        import importlib.util

        def has(mod: str) -> bool:
            try:
                return importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                return False

        def sam_ready() -> bool:
            """SAM พร้อมใช้ไหม = มีโมดูล + มีไฟล์ checkpoint"""
            if not has("segment_anything"):
                return False
            if detection_cfg is None:
                return True
            from .detection.sam_segmenter import SAMSegmenter
            return SAMSegmenter.resolve_checkpoint(detection_cfg.sam_checkpoint) is not None

        # ลำดับความสามารถ: sam (ดีสุด) -> fastsam -> box (ไม่ต้องพึ่งอะไร)
        name = (name or "auto").strip().lower()

        if name == "auto":
            if sam_ready():
                return "sam"
            if has("ultralytics"):
                log.warning("segmenter=auto: ใช้ 'fastsam' (SAM ยังไม่พร้อม — ขาด checkpoint "
                            "หรือโมดูล segment-anything)")
                return "fastsam"
            log.warning("segmenter=auto: ใช้ 'box' (ติดตั้ง SAM: pip install -r requirements-ml.txt "
                        "แล้ว python scripts/download_weights.py --model vit_b)")
            return "box"

        if name == "sam":
            if sam_ready():
                return "sam"
            reason = ("ขาดโมดูล segment-anything" if not has("segment_anything")
                      else "ยังไม่มีไฟล์ checkpoint ของ SAM")
            if allow_fallback:
                log.warning(
                    "segmenter 'sam' ยังไม่พร้อม (%s) — ถอยไปใช้ 'box' แทนชั่วคราว\n"
                    "        จะใช้ SAM ได้ต้อง:\n"
                    "          pip install -r requirements-ml.txt\n"
                    "          python scripts/download_weights.py --model vit_b",
                    reason,
                )
                return "box"
            raise RuntimeError(
                f"segmenter 'sam' ยังไม่พร้อม — {reason}\n"
                f"  ดาวน์โหลด checkpoint: python scripts/download_weights.py --model vit_b"
            )

        need = {"fastsam": "ultralytics", "box": None}.get(name)
        if need is not None and not has(need):
            if allow_fallback:
                log.warning(
                    "segmenter '%s' ยังไม่พร้อม (ขาดโมดูล '%s') — ถอยไปใช้ 'box' แทนชั่วคราว\n"
                    "        ถ้าต้องการ '%s': pip install -r requirements-ml.txt",
                    name, need, name,
                )
                return "box"
            raise RuntimeError(
                f"segmenter '{name}' ยังไม่พร้อม — ขาดโมดูล '{need}'\n"
                f"  ติดตั้งด้วย: pip install -r requirements-ml.txt"
            )
        return name

    def _default_detection(self) -> DetectionPipeline:
        d = self.cfg.detection
        engine = (d.ocr_engine or "easyocr").strip().lower()
        if engine == "tesseract":
            from .detection.ocr_detector import TesseractDetector
            detector = TesseractDetector(d)
        elif engine == "paddleocr":
            from .detection.paddle_detector import PaddleOCRDetector
            detector = PaddleOCRDetector(d, device=d.sam_device)
        else:
            detector = EasyOCRDetector(d, gpu=d.sam_device.startswith("cuda"))

        # เลือก segmenter ตาม config (sam / fastsam / box / auto) โดยไม่ hardcode SAM
        name = self.resolve_segmenter_name(d.segmenter, allow_fallback=d.auto_fallback,
                                           detection_cfg=d)
        from .detection.sam_segmenter import BoxPromptSegmenter, FastSAMSegmenter, SAMSegmenter

        segmenter = {"sam": SAMSegmenter, "fastsam": FastSAMSegmenter,
                     "box": BoxPromptSegmenter}[name](d)
        return DetectionPipeline(detector, segmenter, d)

    @staticmethod
    def resolve_inpainter_method(name: str, allow_fallback: bool = True) -> str:
        """
        แปลงชื่อวิธี inpaint จาก config ให้เป็นตัวที่ "รันได้จริง" บนเครื่องนี้

            "auto" -> เลือกตัวที่ดีที่สุดที่ติดตั้งไว้ (lama -> iopaint -> opencv -> median)
            อื่น ๆ -> ตรวจ dependency จริง ถ้าไม่พร้อมและ auto_fallback=True
                      จะถอยไปใช้ตัวที่รันได้แทน (พร้อมเตือน) เพื่อไม่ให้ pipeline ล้ม

        เหตุผล: `simple-lama-inpainting` กำหนด pillow<10 / numpy<2 จึงติดตั้งไม่ได้
        บน Python 3.13/3.14 — ต้องมีทางถอยไปใช้ OpenCV/Median ให้รันต่อได้
        """
        import importlib.util

        def has(mod: str) -> bool:
            try:
                return importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                return False

        needs = {
            "lama": None,            # เป็น "กลุ่มนามธรรม" -> แก้ด้านล่าง (simple หรือ torch)
            "lama-simple": "simple_lama_inpainting",
            "lama-torch": "torch",
            "iopaint": "iopaint",
            "opencv": "cv2",
            "median": None,          # ไม่ต้องพึ่งอะไร (ใช้ numpy ล้วน)
        }
        name = (name or "auto").strip().lower()

        # ชื่อโมเดลของ IOPaint ที่รู้จัก (ใช้เมื่อผู้ใช้ตั้ง method เป็นชื่อโมเดลโดยตรง)
        iopaint_models = {"mat", "migan", "zits", "ldm", "manga", "fcf", "cv2"}

        # ตัวช่วยเลือก LaMa ที่ "ใช้ได้จริง": ถ้ามี simple-lama-inpainting ใช้ตัวนั้น
        # ไม่งั้นถ้ามี torch -> ใช้ lama-torch (โหลด big-lama.pt ตรง ๆ ใช้ได้บน Python 3.13+)
        def pick_lama() -> str:
            if has("simple_lama_inpainting"):
                return "lama-simple"
            if has("torch"):
                return "lama-torch"
            return "median"

        if name == "auto":
            for cand in ("lama", "iopaint", "opencv", "median"):
                if cand == "lama":
                    return pick_lama()
                need = needs[cand]
                if need is None or has(need):
                    return cand
            return "median"

        if name == "lama":
            return pick_lama()

        # ตั้ง method เป็นชื่อโมเดลของ IOPaint โดยตรง (เช่น mat/migan/zits)
        if name in iopaint_models:
            if has("iopaint"):
                return name
            if allow_fallback:
                log.warning("inpainting.method '%s' ต้องใช้ IOPaint (ยังไม่ติดตั้ง) — "
                            "ถอยไปใช้ '%s'", name, pick_lama())
                return pick_lama()
            raise RuntimeError("ต้องติดตั้ง IOPaint: pip install iopaint")

        # ตรวจก่อนว่าเป็นชื่อที่ "รู้จัก" ไหม — ถ้าไม่ ให้ error ชัดเจน (ไม่ส่งต่อไป IOPaint มั่ว ๆ)
        need = needs.get(name)
        if need is None and name not in needs:
            raise ValueError(
                f"ไม่รู้จัก inpainting.method '{name}'\n"
                f"  ที่รองรับ: lama | lama-torch | lama-simple | iopaint | opencv | median | auto\n"
                f"  (หรือชื่อโมเดลของ IOPaint เช่น: {', '.join(sorted(iopaint_models))})"
            )

        if need is not None and not has(need):
            if allow_fallback:
                logger_name = "median" if name in ("lama-simple", "lama-torch", "iopaint") else "opencv"
                hint = ""
                if name == "lama-torch":
                    hint = "  (ติดตั้ง torch: pip install torch torchvision)"
                elif name == "lama-simple":
                    hint = "  (simple-lama-inpainting ต้องใช้ Python <=3.12)"
                log.warning(
                    "inpainting.method '%s' ยังไม่พร้อม (ขาดโมดูล '%s') — "
                    "ถอยไปใช้ '%s' แทนชั่วคราว%s",
                    name, need, logger_name, hint,
                )
                return logger_name
            raise RuntimeError(
                f"inpainting.method '{name}' ยังไม่พร้อม — ขาดโมดูล '{need}'\n"
                f"  ติดตั้งด้วย: pip install torch && python scripts/download_weights.py --lama\n"
                f"  หรือใช้: SCANLATION_INPAINT_METHOD=median"
            )
        return name

    def _default_inpainter(self) -> Inpainter:
        i = self.cfg.inpainting
        method = self.resolve_inpainter_method(i.method,
                                               allow_fallback=getattr(i, "auto_fallback", True))
        if method in ("lama", "lama-simple"):
            return LamaInpainter(i)
        if method == "lama-torch":
            from .inpainting.lama_torch_inpainter import LamaTorchInpainter
            return LamaTorchInpainter(i)
        if method == "opencv":
            from .inpainting.opencv_inpainter import OpenCVInpainter
            return OpenCVInpainter(i)
        if method == "median":
            from .inpainting.opencv_inpainter import MedianFillInpainter
            return MedianFillInpainter(i)
        # ที่เหลือ = iopaint (รองรับชื่อโมเดลอื่น ๆ เช่น mat/migan)
        from .inpainting.lama_inpainter import IOPaintLaMaInpainter
        return IOPaintLaMaInpainter(i, model_name=method)

    def _default_translator(self) -> Translator:
        return LLMTranslator(self.cfg.translation)

    # ------------------------------------------------------------------
    # สเตจ 0: Download & Ingest (ดึงรูปจากเว็บ → บังคับ WebP → input_dir)
    # ------------------------------------------------------------------
    def download(self, chapter_url: str, out_dir: Optional[str] = None) -> DownloadReport:
        """
        ดาวน์โหลดรูปของ chapter หนึ่งจากเว็บ แล้ววางเป็นไฟล์ WebP ใน input_dir
        (พร้อมให้ process_page/run อ่านต่อได้ทันที)

        out_dir : ถ้าระบุ จะเขียนทับ cfg.download.out_dir ชั่วคราว
        """
        if out_dir:
            self.downloader.cfg.out_dir = out_dir
        report = self.downloader.download_chapter(chapter_url)
        log.info("ดาวน์โหลดเสร็จ: %s", report.summary())
        return report

    def download_and_run(self, chapter_url: str, pages: Optional[Sequence[int]] = None,
                         do_translate: bool = True, out_dir: Optional[str] = None) -> List[ScanlationResult]:
        """ทางลัด: ดาวน์โหลด chapter → แปลทันที (ใช้บ่อยที่สุด)"""
        self.download(chapter_url, out_dir=out_dir)
        return self.run(pages=pages, do_translate=do_translate)

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

        # (5) Monitoring — รวบรวม metrics + เขียนรายงานประจำหน้า
        collector = MetricsCollector(self._work_path("."))
        metrics = collector.begin_page(index=index, image_path=str(image_path))
        collector.analyze_detection(metrics, page)
        if do_translate:
            collector.analyze_translation(metrics, page)
        collector.analyze_inpainting(metrics, page, mask)
        collector.analyze_typesetting(metrics, page)
        collector.finalize_page(metrics, page=page, mask=mask, image=image)

        out_path: Optional[str] = None
        if save:
            out_path = str(self._output_path(image_path, index))
            image_io.save_image(final, out_path)
            log.info("บันทึกผลลัพธ์ -> %s", out_path)
            out_path = self._maybe_compress(out_path)

        return ScanlationResult(page_index=index, input_path=str(image_path),
                                output_path=out_path, page=page)

    # ------------------------------------------------------------------
    # บันทึกผลลัพธ์ + บีบอัดให้ไม่เกินขนาดที่กำหนด (ตาม cfg.output)
    # ------------------------------------------------------------------
    def _maybe_compress(self, out_path: str) -> str:
        """
        บีบอัดไฟล์ผลลัพธ์ให้ไม่เกิน cfg.output.target_size_mb (ถ้าเปิดใช้)

        คืนค่า path สุดท้าย (นามสกุลอาจเปลี่ยนตาม cfg.output.format)
        """
        o = self.cfg.output
        if not getattr(o, "compress", False):
            return out_path
        raw_fmt = str(getattr(o, "format", "webp")).strip().lower()
        # "keep" = คงนามสกุล/ฟอร์แมตเดิมของไฟล์ (ไม่บังคับเปลี่ยนเป็น webp)
        fmt = (Path(out_path).suffix.lstrip(".") or "webp") if raw_fmt == "keep" \
            else normalize_format(raw_fmt)
        target = target_bytes_from_mb(getattr(o, "target_size_mb", 1.0))
        try:
            res = compress_image_file(
                out_path, target, fmt=fmt,
                min_quality=int(getattr(o, "min_quality", 40)),
                max_quality=int(getattr(o, "max_quality", 92)),
                allow_downscale=bool(getattr(o, "allow_downscale", True)),
                max_downscale_ratio=float(getattr(o, "max_downscale_ratio", 0.6)),
            )
            if not res.ok:
                log.warning("บีบอัดแล้วยังเกินเป้า %.2f MB (ได้ %s) — ลองลด target หรือย่อขนาดเพิ่ม",
                            o.target_size_mb, human_size(res.size_bytes))
            return str(res.path)
        except Exception as exc:  # noqa: BLE001 - บีบอัดล้มไม่ควรทำให้ทั้ง pipeline ล้ม
            log.warning("บีบอัดผลลัพธ์ไม่สำเร็จ (%s): %s", out_path, exc)
            return out_path

    # ------------------------------------------------------------------
    # ประมวลผลหลายหน้า / ทั้งเรื่อง
    # ------------------------------------------------------------------
    def run(self, pages: Optional[Sequence[int]] = None,
            do_translate: bool = True) -> List[ScanlationResult]:
        """
        ประมวลผลหลายหน้า/ทั้งเรื่อง

        ★ กัน "หน้าที่ล้มเหลวไปทั้งเรื่อง": จับ error ระดับ "หน้า" แล้วเดินต่อ
        เดิมถ้าหน้าใดหน้า 1 ล้ม (เช่น LLM ตอบ JSON ขาด / เน็ตสะดุดจน retry ครบ)
        exception จะลอยขึ้นไปทำให้ "ทุกหน้าหลังจากนั้นไม่ถูกประมวลผลเลย"
        → ผู้ใช้เห็นเป็นบางหน้าไทย บางหน้าอังกฤษ (เหมือนแปลไม่ทั้งเรื่อง)
        """
        paths = self.discover_pages()
        if pages is not None:
            wanted = set(pages)
            paths = [(i, p) for (i, p) in paths if i in wanted]

        results: List[ScanlationResult] = []
        failures: List[tuple[int, str, str]] = []
        for index, path in paths:
            try:
                results.append(self.process_page(path, index=index, do_translate=do_translate))
            except Exception as exc:  # noqa: BLE001 - ตั้งใจกันไม่ให้ทั้งเรื่องล้ม
                log.error("หน้า %d (%s) ล้มเหลว: %s: %s — ข้ามไปหน้าถัดไป",
                          index, path.name, type(exc).__name__, exc)
                failures.append((index, path.name, f"{type(exc).__name__}: {exc}"))

        if failures:
            log.warning("มี %d หน้าที่ประมวลผลไม่สำเร็จ: %s",
                        len(failures), ", ".join(f"หน้า {i}" for i, _, _ in failures))
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
        o = self.cfg.output
        # เลือกนามสกุลตามฟอร์แมตผลลัพธ์ (ถ้า "keep" ให้คง .png เดิม)
        fmt = getattr(o, "format", "webp")
        ext = ".png" if str(fmt).lower() == "keep" else ext_for(fmt)
        return out_dir / f"page-{index:04d}{ext}"

    def _dump_work(self, page: Page, stage: str) -> None:
        try:
            page.save_json(self._work_path(f"{stage}_p{page.index:04d}.json"))
        except Exception as exc:  # noqa: BLE001 - การ dump ไม่ควรทำให้ pipeline ล้ม
            log.warning("บันทึก intermediate ไม่สำเร็จ (%s): %s", stage, exc)
