# -*- coding: utf-8 -*-
"""
scanlation
==========
แพ็กเกจหลักของระบบแปลมังงะระดับ Professional Grade

โครงสร้างโมดูล:
    schemas      : Data model กลาง (BBox / TextBlock / Bubble / Page)
    config       : ค่า config ทั้งหมด (dataclass)
    utils        : ฟังก์ชันช่วยเหลือ (image io, geometry, registry, logging)
    detection    : OCR + SAM segmentation  -> Bubble พร้อม mask
    inpainting   : ลบข้อความเดิมด้วย LaMa / OpenCV
    translation  : แปลด้วย Claude 3.5 Sonnet / GPT-4o (มี character context)
    typesetting  : จัดวางข้อความไทยด้วย Pillow (word wrap + พอดีบอลลูน)
    pipeline     : ตัวเชื่อมทุกสเตจเข้าด้วยกัน (orchestrator)
    cli          : command-line interface
"""

from __future__ import annotations

__version__ = "2.0.0"

from . import providers
from .config import ScanlationConfig
from .env import apply_env, load_dotenv, resolve_api_key
from .schemas import BBox, BlockKind, Bubble, Page, TextBlock

__all__ = [
    "ScanlationConfig",
    "providers",
    "apply_env",
    "load_dotenv",
    "resolve_api_key",
    "BBox",
    "BlockKind",
    "Bubble",
    "Page",
    "TextBlock",
    "__version__",
]
