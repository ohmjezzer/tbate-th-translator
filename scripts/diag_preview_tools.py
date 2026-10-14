#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diag_preview_tools.py
=============================
ทดสอบเครื่องมือในหน้า Preview (ซึ่งก่อนหน้านี้มีบั๊ก):
    1. "เพิ่มบอลลูน" (add bubble) + ตัวเลือกลบข้อความเดิมในกรอบ (erase)
    2. "แปรงลบพื้นหลัง" (inpaint strokes) + undo
    3. "ลบข้อความในกรอบ" (inpaint box) — ใช้เมื่อ detection ตกหล่น
    4. ตรวจว่า mask ที่ได้ตรงกับพิกัดจริง (กันบั๊ก "inpaint ผิดจุด")

รัน:  python scripts/diag_preview_tools.py

ไม่ต้องใช้ SAM/LaMa/LLM จริง — ใช้몯 median inpainter + mock pipeline
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanlation.config import ScanlationConfig          # noqa: E402
from scanlation.pipeline import ScanlationPipeline      # noqa: E402
from scanlation.schemas import BBox, BlockKind, Bubble, Page, TextBlock  # noqa: E402
from scanlation.utils import image_io                   # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, ok: bool, extra: str = "") -> None:
    global PASS, FAIL
    ok = bool(ok)
    PASS += ok
    FAIL += (not ok)
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {extra}" if extra and not ok else ""))


# ---------------------------------------------------------------------------
# mock components (ให้ pipeline เดินครบโดยไม่ต้องมี SAM/LaMa)
# ---------------------------------------------------------------------------
class MockDetection:
    def detect_page(self, image, index=0, image_path=""):
        bbox = BBox.from_list([100, 100, 400, 220])
        blk = TextBlock(uid="t0", bbox=BBox.from_list([120, 120, 380, 200]),
                        source_text="HELLO WORLD", confidence=0.9)
        bub = Bubble(uid="bub-1", bbox=bbox, kind=BlockKind.SPEECH, blocks=[blk])
        return Page(index=index, image_path=image_path,
                    width=image.shape[1], height=image.shape[0], bubbles=[bub])


class MockTranslator:
    def translate_page(self, page):
        for b in page.bubbles:
            b.set_translation("สวัสดีชาวโลก")
        return page


def main() -> int:
    work = Path("/tmp/preview_tools_test")
    shutil.rmtree(work, ignore_errors=True)
    (work / "orig").mkdir(parents=True)

    # ภาพหน้า: พื้นขาว + จุดดำแยกก้อนชัดเจน (เพื่อพิสูจน์ว่าลบ "ตรงตำแหน่ง" จริง)
    img = np.full((1000, 700, 3), 255, np.uint8)
    img[120:200, 120:380] = (20, 20, 20)     # ข้อความก้อนบน (detected)
    img[400:440, 480:540] = (20, 20, 20)     # จุด A: มีไว้ให้ "แปรงลบ" ทดสอบ
    img[400:440, 60:120] = (20, 20, 20)      # จุด B: ตัวควบคุม (ต้องไม่ถูกลบ)
    img[600:660, 300:600] = (20, 20, 20)     # ข้อความก้อนล่าง (missed -> ใช้เพิ่มบอลลูนเอง)
    image_io.save_image(img, work / "orig" / "page-0001.png")

    cfg = ScanlationConfig()
    cfg.paths.input_dir = str(work / "orig")
    cfg.paths.output_dir = str(work / "out")
    cfg.paths.work_dir = str(work / ".scanlation")
    cfg.inpainting.method = "median"          # ไม่ต้องพึ่ง LaMa ในสภาพแวดล้อมทดสอบ
    cfg.inpainting.mask_dilate = 0
    cfg.detection.ocr_engine = "tesseract"    # ไม่ถูกเรียกใช้ (ใช้ mock detection)

    pipe = ScanlationPipeline(cfg, detection=MockDetection(),
                             translator=MockTranslator())
    pipe.process_page(work / "orig" / "page-0001.png", index=1, do_translate=True, save=True)

    from scanlation.web.app import create_app
    app = create_app(cfg)
    c = app.test_client()

    print("=== 1) เพิ่มบอลลูน (add bubble) ===")
    r = c.post("/api/page/1/bubbles", json={
        "op": "add", "bbox": [300, 600, 600, 660], "kind": "sfx"})
    check("POST /bubbles (add) = 200", r.status_code == 200, str(r.get_json()))
    uid = r.get_json().get("uid")
    check("ได้ uid ของบอลลูนใหม่", bool(uid), str(r.get_json()))

    r = c.get("/api/page/1")
    uids = [b["uid"] for b in r.get_json()["bubbles"]]
    check("บอลลูนใหม่ถูกบันทึกจริง", uid in uids, str(uids))

    print("=== 2) เพิ่มบอลลูน + ลบข้อความเดิมในกรอบ (erase) ===")
    r = c.post("/api/page/1/bubbles", json={
        "op": "add", "bbox": [300, 600, 600, 660], "kind": "sfx", "erase": True})
    check("POST /bubbles (add+erase) = 200", r.status_code == 200, str(r.get_json()))
    check("คืนค่า erased=True (ลบสำเร็จ)", r.get_json().get("erased") is True, str(r.get_json()))

    # ตรวจว่าพิกเซลข้อความเดิมในกรอบถูกลบจริง (ส่วนที่เคยดำ ควรสว่างขึ้น)
    cleaned = image_io.load_image_rgb(work / ".scanlation" / "2_cleaned_p0001.png")
    region = cleaned[600:660, 300:600]
    check("ข้อความเดิมในกรอบถูกลบ (บริเวณนั้นสว่างขึ้น)",
          region.mean() > 120, f"mean={region.mean():.1f}")

    print("=== 3) แปรงลบ (inpaint strokes) ต้องตรงตำแหน่ง ===")
    # แปรงคลุม "จุด A" (พิกัดภาพ: x≈510, y≈420) ด้วยรัศมี 30
    r = c.post("/api/page/1/inpaint", json={
        "op": "inpaint", "grow": 2,
        "strokes": [{"points": [[510, 420]], "radius": 30}]})
    check("POST /inpaint (strokes) = 200", r.status_code == 200, str(r.get_json()))
    cleaned2 = image_io.load_image_rgb(work / ".scanlation" / "2_cleaned_p0001.png")
    hitA = cleaned2[405:435, 485:535].mean()     # จุดที่แปรง (จุด A) -> ควรสว่างขึ้น
    ctrlB = cleaned2[405:435, 65:115].mean()     # จุดควบคุม (จุด B) -> ต้องคงมืด
    check("ตำแหน่งที่แปรงถูกลบจริง (จุด A สว่างขึ้น)", hitA > 120, f"A={hitA:.1f}")
    check("ตำแหน่งอื่นไม่ถูกลบ (จุด B ยังมืด = mask ไม่เลื่อน)", ctrlB < 120, f"B={ctrlB:.1f}")

    print("=== 4) ย้อนการลบ (undo) ===")
    r = c.post("/api/page/1/inpaint", json={"op": "undo"})
    check("POST /inpaint (undo) = 200", r.status_code == 200, str(r.get_json()))
    back = image_io.load_image_rgb(work / ".scanlation" / "2_cleaned_p0001.png")
    check("undo คืนภาพก่อนลบ (จุด A กลับมืด)", back[405:435, 485:535].mean() < 120,
          f"A={back[405:435, 485:535].mean():.1f}")

    print("=== 5) ลบข้อความในกรอบ (inpaint box) ===")
    r = c.post("/api/page/1/inpaint", json={
        "op": "box", "bbox": [100, 100, 400, 220], "grow": 2, "text_only": True})
    check("POST /inpaint (box) = 200", r.status_code == 200, str(r.get_json()))
    check("box: คืน erased_px > 0", (r.get_json().get("erased_px") or 0) > 0, str(r.get_json()))

    print("=== 6) ตรวจ UI assets ใหม่ ===")
    for f in ("index.html", "app.js", "style.css"):
        r = c.get(f"/static/{f}")
        check(f"GET /static/{f} = 200", r.status_code == 200)
    html = c.get("/static/index.html").get_data(as_text=True)
    check("มีแถบสถานะ (statusBar)", 'id="statusBar"' in html)
    check("มีม็อดัลช่วยเหลือ (helpModal)", 'id="helpModal"' in html)
    check("มีตัวเลือกลบข้อความเดิม (newErase)", 'id="newErase"' in html)
    css = c.get("/static/style.css").get_data(as_text=True)
    check("แก้ CSS: เปิด pointer-events ในโหมด adding",
          ".stage-inner.adding  .brush-canvas" in css or ".stage-inner.adding .brush-canvas" in css)
    js = c.get("/static/app.js").get_data(as_text=True)
    check("มี toImageXY (แปลงพิกัดด้วยกรอบรูป)", "function toImageXY" in js)
    check("มี eraseStrokeAt (โหมดลบเส้น)", "function eraseStrokeAt" in js)

    print(f"\nผลรวม: ✅ ผ่าน {PASS} | ❌ ไม่ผ่าน {FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
