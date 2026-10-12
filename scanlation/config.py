# -*- coding: utf-8 -*-
"""
config.py
=========
รวมการตั้งค่าทั้งหมดของระบบไว้เป็น dataclass เดียว

หลักการ: "config เป็นข้อมูล ไม่ใช่โค้ด" — ทุกสเตจรับ config ของตัวเองเข้ามา
ทำให้ปรับพฤติกรรมได้โดยไม่ต้องแก้โค้ด และทดสอบ (unit test) ได้ง่าย

รองรับการโหลดทับจากไฟล์ YAML/JSON (ถ้าติดตั้ง PyYAML) และจาก environment variable
(โดยเฉพาะ API key ของผู้ให้บริการ LLM)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# 1) Paths — ตำแหน่งไฟล์/โฟลเดอร์ต่าง ๆ
# ---------------------------------------------------------------------------
@dataclass
class PathConfig:
    input_dir: str = "orig"              # รูปต้นฉบับ
    output_dir: str = "out"              # รูปผลลัพธ์
    work_dir: str = ".scanlation"        # intermediate result (json/png)
    fonts_dir: str = "fonts"             # ฟอนต์
    config_dir: str = "config"           # glossary / characters


# ---------------------------------------------------------------------------
# 2) Detection — การหาและตัดขอบเขตข้อความ (OCR + SAM)
# ---------------------------------------------------------------------------
@dataclass
class DetectionConfig:
    # --- OCR ---
    ocr_engine: str = "easyocr"          # "easyocr" | "tesseract"
    ocr_langs: tuple = ("en",)
    ocr_min_confidence: float = 0.40     # กรองคำที่มั่นใจต่ำ
    merge_gap: int = 16                  # ระยะห่างสูงสุด (px) ที่จัดว่าเป็นบอลลูนเดียวกัน

    # --- SAM (Segment Anything Model) ---
    sam_model_type: str = "vit_h"        # vit_h | vit_l | vit_b  (หรือ mobile_sam)
    sam_checkpoint: str = "weights/sam_vit_h_4b8939.pth"
    sam_device: str = "cuda"             # cuda | cpu | mps
    sam_expand: int = 6                  # ขยายกรอบก่อน prompt ให้ SAM (px)
    sam_mask_fill_ratio: float = 0.45    # mask ต้องกินพื้นที่ >= ratio นี้ จึงนับเป็น "บอลลูน"
    sam_bubble_whiteness: float = 0.72   # ความสว่างเฉลี่ยขั้นต่ำของพื้นที่บอลลูน
    use_auto_mask: bool = False          # ใช้ SAM AutomaticMaskGenerator (ช้ากว่า แม่นกว่า)


# ---------------------------------------------------------------------------
# 3) Inpainting — การลบข้อความเดิมออกจากภาพ
# ---------------------------------------------------------------------------
@dataclass
class InpaintingConfig:
    method: str = "lama"                 # "lama" | "iopaint" | "opencv" | "median"
    device: str = "cuda"                 # อุปกรณ์ของ LaMa
    mask_dilate: int = 8                 # ขยาย mask ก่อน inpaint (px) กันขอบตัวอักษรค้าง
    text_delta: int = 45                 # เกณฑ์ต่างสี (0-255) ที่นับว่าเป็น "ตัวอักษร"
    opencv_radius: int = 4               # รัศมีของ cv2.inpaint (โหมด opencv)
    opencv_method: str = "telea"         # telea | ns
    pad_to_multiple: int = 8             # LaMa ต้องการขนาดหารด้วย 8 ลงตัว


# ---------------------------------------------------------------------------
# 4) Translation — การแปลด้วย LLM
# ---------------------------------------------------------------------------
@dataclass
class TranslationConfig:
    backend: str = "anthropic"           # "anthropic" | "openai"
    model: str = "claude-3-5-sonnet-latest"  # แนะนำ "gpt-4o" สำหรับฝั่ง OpenAI
    temperature: float = 0.3
    max_tokens: int = 4096
    target_lang: str = "Thai"
    glossary_path: str = "config/glossary.json"
    characters_path: str = "config/characters.json"
    cache_path: str = ".scanlation/translation_cache.json"
    max_retries: int = 3
    timeout: float = 120.0


# ---------------------------------------------------------------------------
# 5) Typesetting — การจัดวางข้อความไทย (Pillow)
# ---------------------------------------------------------------------------
@dataclass
class TypesettingConfig:
    regular_font: str = "fonts/Sarabun-Regular.ttf"
    bold_font: str = "fonts/Sarabun-Bold.ttf"
    sfx_font: str = "fonts/Sarabun-Bold.ttf"
    min_font_size: int = 10
    max_font_size: int = 200
    line_spacing: float = 1.28           # ตัวคูณความสูงบรรทัด
    padding_ratio: float = 0.14          # ระยะขอบในบอลลูน (สัดส่วนของฟอนต์)
    sfx_stroke_ratio: float = 0.13       # ความหนาเส้นขอบตัวอักษร SFX
    auto_contrast: bool = True           # ปรับสีตัวอักษรตามความสว่างพื้นหลังอัตโนมัติ
    fit_attempts: int = 2                # จำนวนรอบค้นหาขนาดฟอนต์ (strict → relaxed)


# ---------------------------------------------------------------------------
# 6) รวมทุกอย่าง
# ---------------------------------------------------------------------------
@dataclass
class ScanlationConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    inpainting: InpaintingConfig = field(default_factory=InpaintingConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    typesetting: TypesettingConfig = field(default_factory=TypesettingConfig)

    # ---- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---- loading ---------------------------------------------------------
    @staticmethod
    def load(path: str | Path) -> "ScanlationConfig":
        """โหลด config จาก .json หรือ .yaml แล้ว merge ทับค่า default"""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("ต้องติดตั้ง PyYAML เพื่อโหลดไฟล์ .yaml") from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)

        cfg = ScanlationConfig()
        for section, values in (data or {}).items():
            if not hasattr(cfg, section):
                continue
            target = getattr(cfg, section)
            for k, v in (values or {}).items():
                if hasattr(target, k):
                    setattr(target, k, v)
        return cfg

    # ---- อ่าน API key จาก environment ------------------------------------
    def api_key(self) -> Optional[str]:
        """ดึง API key ของ backend ที่เลือกจาก environment variable"""
        if self.translation.backend == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        if self.translation.backend == "openai":
            return os.environ.get("OPENAI_API_KEY")
        return None
