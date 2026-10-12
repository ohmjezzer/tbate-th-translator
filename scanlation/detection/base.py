# -*- coding: utf-8 -*-
"""
detection/base.py
=================
Interface (abstract base class) ของสเตจ Detection

การแยกเป็น 2 interface ย่อยทำให้ "สลับโมเดลได้อิสระ":
    - TextDetector   : หา "ข้อความ" ที่ไหนบ้าง (จะใช้ EasyOCR / PaddleOCR / Tesseract ก็ได้)
    - BubbleSegmenter: หา "ขอบเขตบอลลูน" สำหรับ mask (จะใช้ SAM / FastSAM / SAM2 ก็ได้)

ทั้งสองต่อกันผ่านข้อมูลกลางใน schemas.py เท่านั้น
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..schemas import Bubble, TextBlock


class TextDetector(ABC):
    """อ่านข้อความต้นฉบับจากภาพ → รายการ TextBlock"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[TextBlock]:
        """image: numpy RGB uint8 (H x W x 3) → List[TextBlock]"""
        raise NotImplementedError


class BubbleSegmenter(ABC):
    """
    สร้าง mask ของพื้นที่ข้อความแต่ละก้อน (บอลลูน/กล่องบรรยาย)
    รวมถึงจัดกลุ่ม TextBlock ให้เป็น Bubble และจำแนกประเภท (speech/narration/sfx)
    """

    @abstractmethod
    def segment(self, image: np.ndarray, text_blocks: List[TextBlock]) -> List[Bubble]:
        """image + TextBlock ที่ตรวจได้ → List[Bubble] ที่มี mask/kind"""
        raise NotImplementedError
