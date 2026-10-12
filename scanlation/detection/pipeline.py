# -*- coding: utf-8 -*-
"""
detection/pipeline.py
=====================
DetectionPipeline — ประกอบ TextDetector + BubbleSegmenter เข้าด้วยกันเป็นสเตจเดียว

    image ──OCR──► TextBlock[] ──SAM──► Bubble[] (มี mask + kind)

ออกแบบให้ "stateless ต่อหน้า": เรียก detect_page(image) แล้วได้ Page กลับมา
จึงรันแบบขนาน (multi-process) ได้ง่ายในอนาคต
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import DetectionConfig
from ..schemas import Page
from ..utils.logging_utils import get_logger
from .base import BubbleSegmenter, TextDetector

log = get_logger(__name__)


class DetectionPipeline:
    def __init__(self, detector: TextDetector, segmenter: BubbleSegmenter,
                 config: Optional[DetectionConfig] = None):
        self.detector = detector
        self.segmenter = segmenter
        self.cfg = config or DetectionConfig()

    def detect_page(self, image: np.ndarray, index: int = 0, image_path: str = "") -> Page:
        """ประมวลผลภาพ 1 หน้า -> Page ที่มี Bubble พร้อม mask"""
        h, w = image.shape[:2]

        # (1) อ่านข้อความด้วย OCR
        text_blocks = self.detector.detect(image)
        log.info("หน้า %d: OCR พบ %d บล็อกข้อความ", index, len(text_blocks))

        # (2) จัดกลุ่ม + ตัด mask + จำแนกประเภท ด้วย segmenter (SAM)
        bubbles = self.segmenter.segment(image, text_blocks)
        log.info("หน้า %d: ได้ %d บอลลูน/พื้นที่ข้อความ", index, len(bubbles))

        return Page(index=index, image_path=image_path, width=w, height=h, bubbles=bubbles)
