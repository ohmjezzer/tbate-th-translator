#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_monitoring.py
==================
สคริปต์ทดสอบระบบ Monitoring แบบครบวงจร (ไม่ต้องใช้ SAM/LaMa/LLM จริง)

สิ่งที่จำลอง:
    1. หน้า A (ปกติ)      — detection เจอ 2 บอลลูน, แปลครบ, mask ปกติ → ควร ✅ ไม่มี issue
    2. หน้า B (มีปัญหา)   — บอลลูนไม่มีคำแปล + คำแปลไทยปนอังกฤษ + mask ใหญ่ผิดปกติ
                            + บอลลูนบวมเกิน 60% ของหน้า → ควรเจอ issues หลายรหัส
    3. Flask API          — ทดสอบ /api/metrics, /api/metrics/<i>, /monitoring/<i>/<file>

วิธีรัน (จากราก repo):
    python scripts/diag_monitoring.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanlation.config import ScanlationConfig
from scanlation.monitoring import MetricsCollector
from scanlation.schemas import BBox, BlockKind, Bubble, Page, TextBlock

PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def make_page(index: int, w: int = 800, h: int = 1200) -> Page:
    return Page(index=index, image_path=f"orig/page-{index:04d}.png", width=w, height=h)


def add_bubble(page: Page, uid: str, box, kind, src: str, th: str = "", conf: float = 0.9):
    bbox = BBox.from_list(box)
    b = Bubble(uid=uid, bbox=bbox, kind=kind,
               blocks=[TextBlock(uid=f"{uid}-t0", bbox=bbox, source_text=src, confidence=conf)])
    b.set_translation(th)
    page.bubbles.append(b)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="scanlation_mon_"))
    work = tmp / ".scanlation"
    work.mkdir(parents=True)
    print(f"📁 work_dir ชั่วคราว: {work}")

    collector = MetricsCollector(work)

    # ---------- หน้า A: ปกติ ----------
    print("\n=== หน้า A (ปกติ — คาดว่าไม่มี issue) ===")
    page_a = make_page(1)
    add_bubble(page_a, "bub-1", [100, 100, 400, 250], BlockKind.SPEECH, "I will protect everyone!", "ฉันจะปกป้องทุกคน!")
    add_bubble(page_a, "bub-2", [100, 400, 400, 520], BlockKind.NARRATION, "The next morning...", "เช้าวันรุ่งขึ้น...")

    mask_a = np.zeros((1200, 800), dtype=np.uint8)
    mask_a[110:240, 110:390] = 255  # ใต้บอลลูน 1
    mask_a[410:510, 110:390] = 255  # ใต้บอลลูน 2

    img_a = np.full((1200, 800, 3), 220, dtype=np.uint8)

    m = collector.begin_page(1, "orig/page-0001.png")
    collector.analyze_detection(m, page_a)
    collector.analyze_translation(m, page_a)
    collector.analyze_inpainting(m, page_a, mask_a)
    collector.analyze_typesetting(m, page_a)
    collector.finalize_page(m, page=page_a, mask=mask_a, image=img_a)

    check("หน้า A: ไม่มี issue", len(m.issues) == 0, f"เจอ {m.issues}")
    check("หน้า A: detection.text_blocks = 2", m.stages["detection"]["text_blocks"] == 2)
    check("หน้า A: translation.translated = 2", m.stages["translation"]["translated"] == 2)
    check("หน้า A: มี artifacts.mask", "mask" in m.artifacts)
    check("หน้า A: มี artifacts.overlay", "overlay" in m.artifacts)
    check("หน้า A: เขียนไฟล์ JSON สำเร็จ", (work / "monitoring" / "page-0001.json").exists())
    check("หน้า A: เขียนไฟล์ mask PNG", (work / "monitoring" / m.artifacts["mask"]).exists())

    # ---------- หน้า B: มีปัญหาครบทุกสเตจ ----------
    print("\n=== หน้า B (ปัญหาครบ — คาดว่าเจอ issues หลายรหัส) ===")
    page_b = make_page(2)
    # บอลลูนบวมเกิน 60% ของภาพ
    add_bubble(page_b, "bub-big", [10, 10, 790, 1100], BlockKind.UNKNOWN, "garbled", "", conf=0.9)
    # บอลลูนแปลไม่ครบ (ไม่มีคำแปล)
    add_bubble(page_b, "bub-miss", [50, 1130, 300, 1180], BlockKind.SPEECH, "Wait!", "", conf=0.85)
    # บอลลูนไทยปนอังกฤษ + OCR มั่นใจต่ำ
    add_bubble(page_b, "bub-mixed", [350, 1130, 600, 1180], BlockKind.SPEECH,
               "The mana core is unstable", "มานา Core ไม่เสถียร", conf=0.30)

    # mask ครอบคลุม 45% ของภาพ (เกินเกณฑ์ 35%)
    mask_b = np.zeros((1200, 800), dtype=np.uint8)
    mask_b[:540, :] = 255

    m_b = collector.begin_page(2, "orig/page-0002.png")
    collector.analyze_detection(m_b, page_b)
    collector.analyze_translation(m_b, page_b)
    collector.analyze_inpainting(m_b, page_b, mask_b)
    collector.analyze_typesetting(m_b, page_b)
    collector.finalize_page(m_b, page=page_b, mask=mask_b)

    codes = {i["code"] for i in m_b.issues}
    print(f"  พบ issue codes: {sorted(codes)}")
    check("หน้า B: เจอ OVERSIZED_BUBBLE", "OVERSIZED_BUBBLE" in codes)
    check("หน้า B: เจอ LOW_CONFIDENCE", "LOW_CONFIDENCE" in codes)
    check("หน้า B: เจอ MISSING_TRANSLATION", "MISSING_TRANSLATION" in codes)
    check("หน้า B: เจอ LEFTOVER_ENGLISH", "LEFTOVER_ENGLISH" in codes)
    check("หน้า B: เจอ HIGH_MASK_COVERAGE", "HIGH_MASK_COVERAGE" in codes)
    check("หน้า B: เจอ UNRENDERED_BUBBLES", "UNRENDERED_BUBBLES" in codes)
    check("หน้า B: บอลลูน bub-miss มี meta.untranslated (จาก pipeline เดิมไม่ใส่ แต่ monitor ตรวจได้)",
          "bub-miss" in m_b.stages["translation"]["missing_uids"])

    # ---------- ทดสอบการอ่านกลับ ----------
    print("\n=== การอ่าน metrics กลับ ===")
    loaded = collector.load_page(2)
    check("load_page(2) อ่านกลับได้", loaded is not None and loaded.page_index == 2)
    check("load_page(999) = None (ยังไม่มี)", collector.load_page(999) is None)
    summary = collector.summary()
    check("summary() คืนจำนวนหน้า = 2", summary["total_pages"] == 2)
    check("summary() รวม issues = 6", summary["total_issues"] == 6, f"ได้ {summary['total_issues']}")
    check("summary() หน้าที่มี issue = 1", summary["pages_with_issues"] == 1)

    # ---------- ทดสอบ Flask API ----------
    print("\n=== Flask API ===")
    try:
        from flask import Flask  # noqa
    except ImportError:
        print("  ⏭ ไม่มี flask — ข้ามส่วน API")
    else:
        from scanlation.web.app import create_app

        cfg = ScanlationConfig()
        cfg.paths.input_dir = str(tmp / "orig")
        cfg.paths.output_dir = str(tmp / "out")
        cfg.paths.work_dir = str(work)
        app = create_app(cfg)
        c = app.test_client()

        r = c.get("/api/metrics")
        check("GET /api/metrics = 200", r.status_code == 200)
        data = r.get_json()
        check("  summary มีครบ 2 หน้า", data.get("total_pages") == 2)

        r = c.get("/api/metrics/2")
        check("GET /api/metrics/2 = 200", r.status_code == 200)
        check("  หน้า 2 มี issues 6 รายการ", len(r.get_json().get("issues", [])) == 6)

        r = c.get("/api/metrics/999")
        check("GET /api/metrics/999 = 404", r.status_code == 404)

        r = c.get("/monitoring/2/page-0002-mask.png")
        check("GET /monitoring/2/page-0002-mask.png = 200", r.status_code == 200)

        # path traversal ต้องโดนบล็อก
        r = c.get("/monitoring/2/..%2F..%2Fetc%2Fpasswd")
        check("path traversal ถูกบล็อก (404/400)", r.status_code in (400, 404))

    print(f"\n{'='*50}\nผลรวม: ✅ ผ่าน {PASS} | ❌ ไม่ผ่าน {FAIL}\n{'='*50}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
