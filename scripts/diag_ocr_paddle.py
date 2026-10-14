#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diag_ocr_paddle.py
==========================
ทดสอบ PaddleOCRDetector โดยไม่ต้องติดตั้ง paddle จริง
(สร้างโมดูล paddleocr ปลอมใน sys.modules แล้วตรวจพฤติกรรม)

    python scripts/diag_ocr_paddle.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    ok = bool(cond)
    PASS += ok
    FAIL += (not ok)
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {extra}" if extra and not ok else ""))


# ---------------------------------------------------------------------------
# 1) Mock โมดูล paddleocr (รูปแบบผลลัพธ์แบบ PaddleX: list[dict])
# ---------------------------------------------------------------------------
def install_fake_paddleocr(style: str = "paddlex") -> None:
    """สร้างโมดูล paddleocr ปลอมที่คืนผลตามรูปแบบที่เลือก"""

    class FakePaddleOCR:
        # ประกาศ signature แบบ v3 เพื่อให้ตัวตรวจ (inspect) มองเห็นพารามิเตอร์
        # ปิด pipeline ย่อยจริง — ไม่เช่นนั้น detector จะถือว่าเป็น v2
        def __init__(self, lang="en", use_doc_orientation_classify=True,
                     use_doc_unwarping=True, use_textline_orientation=True,
                     device="cpu", **kwargs):
            self.kwargs = {
                "lang": lang,
                "use_doc_orientation_classify": use_doc_orientation_classify,
                "use_doc_unwarping": use_doc_unwarping,
                "use_textline_orientation": use_textline_orientation,
                "device": device,
                **kwargs,
            }

        # ผลลัพธ์รูปแบบ PaddleX/v3: list[dict-like]
        _paddlex = [{
            "rec_texts": ["I WILL PROTECT", "EVERYONE I LOVE!", "CRACK!"],
            "rec_scores": [0.97, 0.95, 0.44],
            "rec_polys": [
                np.array([[100, 200], [300, 200], [300, 240], [100, 240]]),
                np.array([[100, 250], [320, 250], [320, 290], [100, 290]]),
                np.array([[500, 700], [650, 700], [650, 760], [500, 760]]),
            ],
        }]
        # ผลลัพธ์รูปแบบ v2 legacy: [ [quad, (text, conf)], ... ]
        _legacy = [
            [np.array([[100, 200], [300, 200], [300, 240], [100, 240]]), ("I WILL PROTECT", 0.97)],
            [np.array([[100, 250], [320, 250], [320, 290], [100, 290]]), ("EVERYONE I LOVE!", 0.95)],
            [np.array([[500, 700], [650, 700], [650, 760], [500, 760]]), ("CRACK!", 0.44)],
        ]

        def predict(self, image):
            # v3 คืน list[dict] เสมอ
            return list(self._paddlex)

        def ocr(self, image, **kw):
            # v2 คืน [ [item, ...] ] (ซ้อน 1 ชั้นต่อหน้า)
            return [list(self._legacy)]

    mod = types.ModuleType("paddleocr")
    mod.PaddleOCR = FakePaddleOCR
    sys.modules["paddleocr"] = mod


# ---------------------------------------------------------------------------
# 2) เทสต์หลัก
# ---------------------------------------------------------------------------
def main() -> int:
    global PASS, FAIL
    from scanlation.config import DetectionConfig
    from scanlation.detection.paddle_detector import PaddleOCRDetector

    img = np.zeros((1200, 800, 3), dtype=np.uint8)

    print("=== รูปแบบผลลัพธ์: PaddleX (dict) ===")
    install_fake_paddleocr("paddlex")
    cfg = DetectionConfig(ocr_min_confidence=0.40)
    det = PaddleOCRDetector(cfg, device="cpu")
    blocks = det.detect(img)
    check("ได้ 3 บล็อก", len(blocks) == 3, f"ได้ {len(blocks)}")
    check("ข้อความถูกต้อง", [b.source_text for b in blocks] ==
          ["I WILL PROTECT", "EVERYONE I LOVE!", "CRACK!"])
    check("confidence ถูกต้อง", [round(b.confidence, 2) for b in blocks] == [0.97, 0.95, 0.44])
    check("bbox ของบล็อกแรกถูกต้อง",
          blocks[0].bbox.to_int() == (100, 200, 300, 240), str(blocks[0].bbox.to_int()))
    check("CRACK! (conf 0.44) ผ่านเกณฑ์ 0.40",
          any(b.source_text == "CRACK!" for b in blocks))

    print("=== รูปแบบผลลัพธ์: Legacy (list ซ้อน) ===")
    install_fake_paddleocr("legacy")
    det2 = PaddleOCRDetector(cfg, device="cpu")
    blocks2 = det2.detect(img)
    check("legacy: ได้ 3 บล็อก", len(blocks2) == 3)
    check("legacy: ข้อความถูกต้อง", [b.source_text for b in blocks2] ==
          ["I WILL PROTECT", "EVERYONE I LOVE!", "CRACK!"])

    print("=== กรองตาม min_confidence ===")
    cfg_strict = DetectionConfig(ocr_min_confidence=0.50)
    det3 = PaddleOCRDetector(cfg_strict, device="cpu")
    blocks3 = det3.detect(img)
    check("conf 0.44 ถูกกรองออกเมื่อเกณฑ์ 0.50", len(blocks3) == 2)

    print("=== ตรวจ constructor ส่งค่าไปยัง paddleocr ===")
    fake = sys.modules["paddleocr"]
    det._ensure_ocr()
    used = det._ocr.kwargs
    check("ปิด doc_orientation_classify", used.get("use_doc_orientation_classify") is False)
    check("ปิด doc_unwarping", used.get("use_doc_unwarping") is False)
    check("ตั้ง lang จาก config", used.get("lang") == "en")

    print("=== ไม่มีโมดูล paddleocr -> ข้อความ error ชัดเจน ===")
    saved = sys.modules.pop("paddleocr")

    class Blocker:
        def find_module(self, name, path=None):
            if name == "paddleocr":
                return self
            return None

        def load_module(self, name):
            raise ImportError("blocked for test")

    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        det4 = PaddleOCRDetector(DetectionConfig(), device="cpu")
        try:
            det4.detect(img)
            check("ต้อง raise RuntimeError", False)
        except RuntimeError as e:
            check("raise RuntimeError พร้อมวิธีติดตั้ง", "pip install paddlepaddle paddleocr" in str(e))
    finally:
        sys.meta_path.remove(blocker)
        sys.modules["paddleocr"] = saved

    print(f"\nผลรวม: ✅ ผ่าน {PASS} | ❌ ไม่ผ่าน {FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
