# -*- coding: utf-8 -*-
"""
schemas.py
==========
โครงสร้างข้อมูลกลาง (Core Data Schemas) ที่ทุกโมดูลในโปรเจกต์ใช้ร่วมกัน

แนวคิดการออกแบบ (Design rationale):
    - แยก "ข้อมูล" ออกจาก "ตรรกะ" อย่างชัดเจน ทุกสเตจ (Detection / Inpainting /
      Translation / Typesetting) จะอ่าน-เขียนผ่าน schema เหล่านี้เท่านั้น
    - ทุกคลาส serialize เป็น JSON ได้ (to_dict / from_dict) เพื่อให้บันทึก
      intermediate result ไว้ตรวจสอบ หรือทำ pipeline ให้ resume ได้
    - ฟิลด์ที่เป็น numpy array (mask, polygon) จะไม่ถูก serialize ลง JSON โดยตรง
      แต่จะเก็บแยกเป็นไฟล์ .png / .json ใน work_dir แทน

ลำดับชั้นความสัมพันธ์:
    Page 1..* Bubble 1..* TextBlock
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Enum ของ "ประเภท" ของกล่องข้อความในมังงะ
# ---------------------------------------------------------------------------
class BlockKind(str, Enum):
    """ประเภทของข้อความ ช่วยให้ typesetter เลือกสไตล์การวาดได้ถูกต้อง"""

    SPEECH = "speech"        # บอลลูนคำพูด (วงกลม/วงรี ขอบมน)
    NARRATION = "narration"  # กล่องบรรยาย (สี่เหลี่ยม)
    SFX = "sfx"              # เสียงเอฟเฟกต์ (ตัวอักษรวาดอิสระ ไม่มีบอลลูน)
    CREDIT = "credit"        # เครดิตสแกนเลชัน / ลายน้ำ
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# BBox: กรอบสี่เหลี่ยม
# ---------------------------------------------------------------------------
@dataclass
class BBox:
    """กรอบสี่เหลี่ยม (พิกัดพิกเซล มุมซ้ายบนเป็นจุดกำเนิด)"""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        # บังคับให้พิกัดเป็น float ของ Python จริง ๆ
        # (EasyOCR/NumPy คืนค่าเป็น np.int32/np.float32 ซึ่ง json.dumps ไม่รู้จัก)
        self.x0 = float(self.x0)
        self.y0 = float(self.y0)
        self.x1 = float(self.x1)
        self.y1 = float(self.y1)

    # ---- คุณสมบัติที่คำนวณได้ -------------------------------------------
    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    # ---- การแปลงค่ารูปแบบต่าง ๆ ------------------------------------------
    def expand(self, px: float) -> "BBox":
        """ขยายกรอบออกทุกด้านตามระยะ px (ค่าลบ = หดเข้า)"""
        return BBox(self.x0 - px, self.y0 - px, self.x1 + px, self.y1 + px)

    def clip(self, width: int, height: int) -> "BBox":
        """บังคับให้กรอบอยู่ในขอบเขตภาพ (width x height)"""
        return BBox(
            max(0.0, min(self.x0, width)),
            max(0.0, min(self.y0, height)),
            max(0.0, min(self.x1, width)),
            max(0.0, min(self.y1, height)),
        )

    def to_int(self) -> Tuple[int, int, int, int]:
        return int(round(self.x0)), int(round(self.y0)), int(round(self.x1)), int(round(self.y1))

    def as_list(self) -> List[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @staticmethod
    def from_list(v: Sequence[float]) -> "BBox":
        return BBox(float(v[0]), float(v[1]), float(v[2]), float(v[3]))

    def union(self, other: "BBox") -> "BBox":
        """รวมสองกรอบเข้าด้วยกัน (ใช้ตอนจัดกลุ่มบรรทัดให้เป็นบอลลูนเดียว)"""
        return BBox(
            min(self.x0, other.x0), min(self.y0, other.y0),
            max(self.x1, other.x1), max(self.y1, other.y1),
        )

    def iou(self, other: "BBox") -> float:
        """Intersection over Union ใช้วัดการทับซ้อน"""
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        return inter / (self.area + other.area - inter + 1e-9)


# ---------------------------------------------------------------------------
# TextBlock: ข้อความหนึ่งรายการที่ OCR อ่านได้ (อาจเป็นบรรทัด)
# ---------------------------------------------------------------------------
@dataclass
class TextBlock:
    """ข้อความต้นฉบับ 1 รายการ พร้อมพิกัดและความมั่นใจจาก OCR"""

    uid: str
    bbox: BBox
    source_text: str = ""          # ข้อความต้นฉบับ (อังกฤษ/จีน/ฯลฯ)
    confidence: float = 0.0        # ค่าความมั่นใจจาก OCR (0..1)
    kind: BlockKind = BlockKind.UNKNOWN
    translation: str = ""          # คำแปล (เติมทีหลังในสเตจ translation)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "bbox": self.bbox.as_list(),
            "source_text": self.source_text,
            "confidence": round(self.confidence, 4),
            "kind": self.kind.value,
            "translation": self.translation,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TextBlock":
        return TextBlock(
            uid=d["uid"],
            bbox=BBox.from_list(d["bbox"]),
            source_text=d.get("source_text", ""),
            confidence=d.get("confidence", 0.0),
            kind=BlockKind(d.get("kind", "unknown")),
            translation=d.get("translation", ""),
            meta=d.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# Bubble: พื้นที่ข้อความหนึ่งก้อน (บอลลูน/กล่องบรรยาย/บริเวณ SFX)
# ---------------------------------------------------------------------------
@dataclass
class Bubble:
    """
    หนึ่งหน่วยของ "พื้นที่ข้อความ" ที่จะถูกลบแล้ววาดคำแปลทับ

    - mask    : binary mask (H x W, dtype bool) ของพื้นที่ที่ต้อง inpaint
                (อาจเป็น None ถ้ายังไม่ผ่านสเตจ segmentation)
    - polygon : จุดขอบของบอลลูน (ใช้วาด/ตรวจสอบ/ตัดข้อความให้อยู่ในขอบ)
    - blocks  : บรรทัดข้อความต้นฉบับที่รวมอยู่ในบอลลูนนี้
    """

    uid: str
    bbox: BBox
    kind: BlockKind = BlockKind.UNKNOWN
    mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    polygon: List[Tuple[int, int]] = field(default_factory=list)
    blocks: List[TextBlock] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- ข้อความ ----------------------------------------------------------
    @property
    def source_text(self) -> str:
        """รวมบรรทัดต้นฉบับเป็นข้อความเดียว (คั่นด้วยช่องว่าง)"""
        return " ".join(b.source_text for b in self.blocks if b.source_text).strip()

    @property
    def translation(self) -> str:
        """คำแปลของบอลลูนนี้ (ดึงจาก TextBlock ตัวแรกที่ถูกเติมคำแปล)"""
        for b in self.blocks:
            if b.translation:
                return b.translation
        return ""

    def set_translation(self, text: str) -> None:
        """กำหนดคำแปลให้บอลลูนนี้ (เก็บที่ TextBlock แรก เพื่อรักษาโครงสร้างเดิม)"""
        if not self.blocks:
            self.blocks.append(TextBlock(uid=f"{self.uid}-t0", bbox=self.bbox))
        self.blocks[0].translation = text

    @property
    def area_px(self) -> int:
        return int(self.bbox.area)

    def to_dict(self, include_mask: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "uid": self.uid,
            "bbox": self.bbox.as_list(),
            "kind": self.kind.value,
            "polygon": self.polygon,
            "blocks": [b.to_dict() for b in self.blocks],
            "meta": self.meta,
        }
        if include_mask and self.mask is not None:
            # mask ถูกเก็บเป็น RLE แบบง่าย (run-length) เพื่อให้ JSON เล็กลง
            d["mask_rle"] = encode_rle(self.mask)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Bubble":
        b = Bubble(
            uid=d["uid"],
            bbox=BBox.from_list(d["bbox"]),
            kind=BlockKind(d.get("kind", "unknown")),
            polygon=[tuple(p) for p in d.get("polygon", [])],
            blocks=[TextBlock.from_dict(x) for x in d.get("blocks", [])],
            meta=d.get("meta", {}),
        )
        if d.get("mask_rle"):
            b.mask = decode_rle(d["mask_rle"])
        return b


# ---------------------------------------------------------------------------
# Page: หนึ่งหน้าของมังงะ
# ---------------------------------------------------------------------------
@dataclass
class Page:
    """ข้อมูลครบของมังงะ 1 หน้า"""

    index: int
    image_path: str
    width: int = 0
    height: int = 0
    bubbles: List[Bubble] = field(default_factory=list)

    def to_dict(self, include_mask: bool = False) -> Dict[str, Any]:
        return {
            "index": self.index,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "bubbles": [b.to_dict(include_mask=include_mask) for b in self.bubbles],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Page":
        return Page(
            index=d["index"],
            image_path=d["image_path"],
            width=d.get("width", 0),
            height=d.get("height", 0),
            bubbles=[Bubble.from_dict(x) for x in d.get("bubbles", [])],
        )

    # ---- Persistence -----------------------------------------------------
    def save_json(self, path: str | Path, include_mask: bool = True) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(include_mask=include_mask), ensure_ascii=False,
                       indent=1, default=_json_default),
            encoding="utf-8",
        )

    @staticmethod
    def load_json(path: str | Path) -> "Page":
        return Page.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Helper: แปลงชนิดข้อมูลที่ json.dumps ไม่รู้จัก (เช่น numpy scalar/array)
# ---------------------------------------------------------------------------
def _json_default(o: Any) -> Any:
    """
    ตัวช่วยให้ json.dumps serialize ค่าที่มาจาก numpy ได้
    (EasyOCR คืนพิกัดเป็น np.int32/np.float32 — ทำให้บันทึก JSON ล้มเหลวถ้าไม่แปลง)
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    # สุดท้าย: แปลงเป็น string เพื่อไม่ให้ทั้งการบันทึกล้ม
    return str(o)


# ---------------------------------------------------------------------------
# Helper: RLE encoding สำหรับเก็บ binary mask ใน JSON
# ---------------------------------------------------------------------------
def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    """บีบอัด binary mask เป็น run-length encoding (เก็บเฉพาะค่า True)"""
    flat = mask.reshape(-1).astype(np.uint8)
    # หาจุดเปลี่ยนค่า 0<->1
    idx = np.flatnonzero(np.diff(flat)) + 1
    starts = np.concatenate(([0], idx))
    lengths = np.diff(np.concatenate((starts, [flat.size])))
    # เก็บเฉพาะ run ที่ค่าเป็น 1 (True)
    runs = [(int(s), int(l)) for s, l in zip(starts, lengths) if flat[s] == 1]
    return {"shape": list(mask.shape), "runs": runs}


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """ถอดรหัส RLE กลับเป็น binary mask"""
    h, w = rle["shape"]
    flat = np.zeros(h * w, dtype=bool)
    for start, length in rle["runs"]:
        flat[start:start + length] = True
    return flat.reshape(h, w)
