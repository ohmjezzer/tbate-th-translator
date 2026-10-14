# -*- coding: utf-8 -*-
"""สเตจ Detection & Masking: OCR หาข้อความ + SAM ตัดขอบเขตบอลลูน"""

from .base import BubbleSegmenter, TextDetector
from .ocr_detector import EasyOCRDetector, TesseractDetector
from .paddle_detector import PaddleOCRDetector
from .sam_segmenter import SAMSegmenter, FastSAMSegmenter, BoxPromptSegmenter
from .pipeline import DetectionPipeline

__all__ = [
    "TextDetector",
    "BubbleSegmenter",
    "EasyOCRDetector",
    "TesseractDetector",
    "PaddleOCRDetector",
    "SAMSegmenter",
    "FastSAMSegmenter",
    "BoxPromptSegmenter",
    "DetectionPipeline",
]
