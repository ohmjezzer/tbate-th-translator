# -*- coding: utf-8 -*-
"""
detection/ocr_detector.py
=========================
ตัวอ่านข้อความ (TextDetector) 2 แบบ:
    - EasyOCRDetector   : ค่าเริ่มต้นของโปรเจกต์ (แม่นกว่า Tesseract กับคอมมิก)
    - TesseractDetector : สำหรับโปรเจกต์เดิมที่ใช้ pytesseract (backward compatible)

ทั้งคู่ทำหน้าที่เหมือนกันเป๊ะ: image -> List[TextBlock]
จึงสลับกันได้โดยไม่กระทบสเตจอื่น (Liskov substitution)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import DetectionConfig
from ..schemas import BBox, BlockKind, TextBlock
from ..utils.logging_utils import get_logger
from ..utils.registry import register
from .base import TextDetector

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------
@register("detector", "easyocr")
class EasyOCRDetector(TextDetector):
    """
    ตัวอ่านข้อความด้วย EasyOCR (โหลดโมเดลแบบ lazy — โหลดจริงเมื่อเรียกครั้งแรก)

    จุดเด่น: จัดกลุ่มข้อความตาม (block, paragraph, line) ที่ EasyOCR คืนมาให้
    จึงแยก "บรรทัด" ได้ตามธรรมชาติ และมีความมั่นใจต่อบรรทัด
    """

    def __init__(self, config: Optional[DetectionConfig] = None, gpu: bool = True):
        self.cfg = config or DetectionConfig()
        self.gpu = gpu
        self._reader = None  # lazy init

    def _ensure_reader(self):
        if self._reader is None:
            import easyocr  # import ตอนใช้งานเพื่อลดเวลา start

            log.info("กำลังโหลด EasyOCR (langs=%s, gpu=%s)...", self.cfg.ocr_langs, self.gpu)
            self._reader = easyocr.Reader(list(self.cfg.ocr_langs), gpu=self.gpu)
        return self._reader

    def detect(self, image: np.ndarray) -> List[TextBlock]:
        reader = self._ensure_reader()
        # detail=1 -> คืน [bbox(4 จุด), text, confidence]
        results = reader.readtext(image, detail=1, paragraph=False)

        blocks: List[TextBlock] = []
        for i, (quad, text, conf) in enumerate(results):
            text = (text or "").strip()
            if not text or conf < self.cfg.ocr_min_confidence:
                continue
            # แปลง quad (4 จุด) -> BBox ตั้งตรง
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            bbox = BBox(min(xs), min(ys), max(xs), max(ys))
            blocks.append(
                TextBlock(uid=f"tb{i}", bbox=bbox, source_text=text,
                          confidence=float(conf), kind=BlockKind.UNKNOWN)
            )
        log.debug("EasyOCR พบ %d บล็อกข้อความ", len(blocks))
        return blocks


# ---------------------------------------------------------------------------
# Tesseract (backward compatible กับโค้ดเดิม extract2.py)
# ---------------------------------------------------------------------------
@register("detector", "tesseract")
class TesseractDetector(TextDetector):
    """
    ตัวอ่านข้อความด้วย Tesseract — เลียนแบบตรรกะจาก extract2.py เดิม
    (ใช้ psm 3 + 11 แล้ว merge บล็อกที่ทับกัน)
    """

    def __init__(self, config: Optional[DetectionConfig] = None, psm_modes=(3, 11)):
        self.cfg = config or DetectionConfig()
        self.psm_modes = psm_modes

    def _blocks_psm(self, image: np.ndarray, psm: int) -> List[TextBlock]:
        import collections

        import pytesseract

        df = pytesseract.image_to_data(
            image, lang="eng", config=f"--psm {psm}", output_type=pytesseract.Output.DATAFRAME
        )
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str)
        df = df[df["text"].str.strip() != ""]

        groups: dict = collections.OrderedDict()
        for _, r in df.iterrows():
            conf = float(r["conf"]) / 100.0
            if conf < self.cfg.ocr_min_confidence:
                continue
            groups.setdefault(int(r["block_num"]), []).append(r)

        out: List[TextBlock] = []
        for bnum, rows in groups.items():
            x0 = int(min(r["left"] for r in rows))
            y0 = int(min(r["top"] for r in rows))
            x1 = int(max(r["left"] + r["width"] for r in rows))
            y1 = int(max(r["top"] + r["height"] for r in rows))
            text = " ".join(str(r["text"]).strip() for r in rows)
            conf = max(float(r["conf"]) / 100.0 for r in rows)
            out.append(TextBlock(uid=f"t{bnum}-{psm}", bbox=BBox(x0, y0, x1, y1),
                                 source_text=text, confidence=conf))
        return out

    def detect(self, image: np.ndarray) -> List[TextBlock]:
        merged: List[TextBlock] = []
        for psm in self.psm_modes:
            for blk in self._blocks_psm(image, psm):
                if any(blk.bbox.iou(m.bbox) > 0.35 for m in merged):
                    continue
                merged.append(blk)
        merged.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
        log.debug("Tesseract พบ %d บล็อกข้อความ", len(merged))
        return merged
