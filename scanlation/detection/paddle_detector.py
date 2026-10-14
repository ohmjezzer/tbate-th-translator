# -*- coding: utf-8 -*-
"""
detection/paddle_detector.py
============================
ตัวอ่านข้อความด้วย **PaddleOCR** (PP-OCRv5) — engine ทางเลือกที่แม่นกว่า EasyOCR

ทำไมต้องมีตัวนี้:
    - EasyOCR ทำงานได้ดีกับตัวอักษรในบอลลูน แต่มักอ่าน SFX (ตัวอักษรแต่งลาย)
      ได้ confidence ต่ำ หรือแยกคำผิดจังหวะ
    - PaddleOCR (โดยเฉพาะรุ่น PP-OCRv5) แม่นกว่าอย่างมีนัยสำคัญบนงานคอมมิก
      ภาษาอังกฤษ ทั้งตัวพิมพ์ใหญ่-เล็ก ฟอนต์ตกแต่ง และตัวอักษรเอียง
    - ฟรี 100% รัน local ได้ ไม่ต้องใช้ API key

การออกแบบ:
    - สืบทอด TextDetector เหมือน EasyOCRDetector/TesseractDetector เป๊ะ
      → pipeline สลับ engine ได้โดยไม่แก้โค้ดส่วนอื่น (Liskov substitution)
    - โหลดโมเดลแบบ lazy (ตอน detect ครั้งแรก) เพื่อไม่ให้คำสั่งอื่น ๆ ช้า
    - รองรับผลลัพธ์ของ paddleocr ทั้ง 2 API รูปแบบ:
        1) รูปแบบใหม่ (paddleocr>=3.0): OCRResult (dict-like) ที่มี
           rec_texts / rec_scores / rec_polys (หรือ dt_polys/rec_boxes)
        2) รูปแบบเก่า (paddleocr 2.x): [[[points], (text, score)], ...]

หมายเหตุด้านเวอร์ชัน:
    - paddleocr>=3.0 ต้องการ paddlepaddle (backend) แยกติดตั้ง:
        pip install paddlepaddle paddleocr
      (บน macOS Apple Silicon ติดตั้งได้ผ่าน pip โดยตรง)
    - ถ้าเครื่องติดตั้งไม่ได้ ระบบจะ error ชัดเจนพร้อมวิธีแก้ (ไม่ล้มเงียบ ๆ)
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

from ..config import DetectionConfig
from ..schemas import BBox, BlockKind, TextBlock
from ..utils.logging_utils import get_logger
from ..utils.registry import register
from .base import TextDetector

log = get_logger(__name__)


# ภาษาของ paddleocr ที่ map จากโค้ดภาษาแบบทั่วไป (ใช้กับ ocr_langs ของ config)
_PADDLE_LANG_MAP = {
    "en": "en",
    "th": "th",
    "ch": "ch",
    "ja": "japan",
    "ko": "korean",
}


@register("detector", "paddleocr")
class PaddleOCRDetector(TextDetector):
    """
    ตัวอ่านข้อความด้วย PaddleOCR (PP-OCRv5)

    พฤติกรรม: image -> List[TextBlock] เหมือน detector ตัวอื่นทุกประการ
    """

    def __init__(self, config: Optional[DetectionConfig] = None, device: str = "cpu"):
        self.cfg = config or DetectionConfig()
        self.device = device
        self._ocr = None  # lazy init

    # ------------------------------------------------------------------
    # โหลดโมเดล (lazy) + แปลงพารามิเตอร์ให้เข้ากับทั้ง v2/v3 API
    # ------------------------------------------------------------------
    def _ensure_ocr(self):
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR  # import ตอนใช้งานจริงเท่านั้น
        except ImportError as exc:
            raise RuntimeError(
                "ยังไม่ได้ติดตั้ง PaddleOCR — ติดตั้งด้วย:\n"
                "    pip install paddlepaddle paddleocr\n"
                "  (หรือสลับกลับไปใช้ easyocr: SCANLATION_OCR_ENGINE=easyocr)"
            ) from exc

        lang = self._resolve_lang()
        # พารามิเตอร์พื้นฐานที่ใช้ได้กับทั้งสองรุ่น
        kwargs: dict = {"lang": lang}

        # ปิด pipeline ย่อยที่ไม่จำเป็นสำหรับมังงะ (เร็วขึ้นมาก) — ใช้ได้เฉพาะ v3
        import inspect

        sig = inspect.signature(PaddleOCR.__init__)
        v3_flags = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        for k, v in v3_flags.items():
            if k in sig.parameters:
                kwargs[k] = v

        # v2 ใช้ use_gpu / v3 ใช้ device
        if "device" in sig.parameters:
            kwargs["device"] = "gpu" if self.device.startswith(("cuda", "gpu")) else "cpu"
        elif "use_gpu" in sig.parameters:
            kwargs["use_gpu"] = self.device.startswith(("cuda", "gpu"))

        log.info("กำลังโหลด PaddleOCR (lang=%s, device=%s)...", lang, self.device)
        self._ocr = PaddleOCR(**kwargs)
        self._api = "v3" if "use_doc_orientation_classify" in sig.parameters else "v2"
        log.info("PaddleOCR พร้อมใช้งาน (API %s)", self._api)
        return self._ocr

    def _resolve_lang(self) -> str:
        langs = list(getattr(self.cfg, "ocr_langs", ("en",)) or ("en",))
        first = langs[0] if langs else "en"
        return _PADDLE_LANG_MAP.get(str(first).lower(), "en")

    # ------------------------------------------------------------------
    # API หลัก
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[TextBlock]:
        ocr = self._ensure_ocr()

        # v3: predict() -> List[OCRResult];  v2: ocr(img, cls=False) -> [[(quad,(txt,score)),...]]
        if getattr(self, "_api", "v3") == "v3":
            results = ocr.predict(image)
            raw = list(self._iter_v3(results))
        else:
            results = ocr.ocr(image, cls=False)
            raw = list(self._iter_v2(results, legacy=True))

        blocks: List[TextBlock] = []
        for i, (bbox, text, conf) in enumerate(raw):
            text = (text or "").strip()
            if not text or conf < self.cfg.ocr_min_confidence:
                continue
            blocks.append(
                TextBlock(uid=f"tb{i}", bbox=bbox, source_text=text,
                          confidence=float(conf), kind=BlockKind.UNKNOWN)
            )
        log.info("PaddleOCR พบ %d บล็อกข้อความ (conf>=%.2f)",
                 len(blocks), self.cfg.ocr_min_confidence)
        return blocks

    # ------------------------------------------------------------------
    # แปลงผลลัพธ์รูปแบบ v3 (OCRResult: dict-like)
    # ------------------------------------------------------------------
    @staticmethod
    def _iter_v3(results: Iterable[Any]) -> Iterable[Tuple[BBox, str, float]]:
        for res in results or []:
            # OCRResult ทำตัวเหมือน dict: มี keys rec_texts/rec_scores/rec_polys
            texts = PaddleOCRDetector._get(res, "rec_texts") or []
            scores = PaddleOCRDetector._get(res, "rec_scores") or []
            polys = (PaddleOCRDetector._get(res, "rec_polys")
                     or PaddleOCRDetector._get(res, "dt_polys")
                     or PaddleOCRDetector._get(res, "rec_boxes") or [])
            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 0.0
                poly = polys[i] if i < len(polys) else None
                if poly is None:
                    continue
                yield PaddleOCRDetector._poly_to_bbox(poly), str(text), score

    # ------------------------------------------------------------------
    # แปลงผลลัพธ์รูปแบบ v2 (list of [[quad], (text, score)])
    # ------------------------------------------------------------------
    @staticmethod
    def _iter_v2(results: Iterable[Any], legacy: bool = False) -> Iterable[Tuple[BBox, str, float]]:
        """
        แปลงผล list ซ้อน — รองรับ 2 รูปแบบ:
            legacy=True : [ [quad, (text, score)], ... ]        (v2 ตรง ๆ)
            legacy=False: [ {rec_texts/rec_scores/rec_polys}, ... ] (v3 ที่ route ผ่าน ocr())
        """
        for page in results or []:
            # รูปแบบ dict-like (OCRResult) → ส่งต่อให้ _iter_v3
            if hasattr(page, "get") or hasattr(page, "json"):
                yield from PaddleOCRDetector._iter_v3([page])
                continue
            for item in page or []:
                try:
                    quad, (text, score) = item[0], item[1]
                except (TypeError, ValueError, IndexError, KeyError):
                    continue
                yield PaddleOCRDetector._poly_to_bbox(quad), str(text), float(score)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get(res: Any, key: str):
        """ดึงค่าจาก OCRResult ซึ่งอาจเป็น dict หรือ object ที่มี .get/.json"""
        if hasattr(res, "get"):
            try:
                return res.get(key)
            except TypeError:
                pass
        data = getattr(res, "json", None)
        if isinstance(data, dict):
            return data.get("res", {}).get(key) or data.get(key)
        return None

    @staticmethod
    def _poly_to_bbox(poly: Any) -> BBox:
        """poly อาจเป็น quad 4 จุด [[x,y]x4] หรือกรอบ [x0,y0,x1,y1]"""
        arr = np.asarray(poly, dtype=float).reshape(-1, 2)
        return BBox(float(arr[:, 0].min()), float(arr[:, 1].min()),
                    float(arr[:, 0].max()), float(arr[:, 1].max()))
