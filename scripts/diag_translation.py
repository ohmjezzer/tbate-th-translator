#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diag_translation.py
===========================
วินิจฉัยสเตจ "Translation" — ทำไมบางบอลลูนยังเป็นไทยปนอังกฤษ (หรือไม่ถูกแปลเลย)

วิธี: จำลอง LLM ด้วย MockBackend ที่เลียนแบบพฤติกรรมจริงของ LLM 6 แบบ
      (ตอบครบ / ขาดบาง id / เก็บคำอังกฤษไว้ / JSON ขาด / ข้อความต้นฉบับว่าง ฯลฯ)
      แล้วดูว่าโค้ดปัจจุบันจัดการแต่ละกรณีอย่างไร

    python scripts/diag_translation.py                  # ทุกกรณี, ผลลัพธ์ที่ diag/trans/
    python scripts/diag_translation.py --scene ok       # เฉพาะกรณีเดียว
    python scripts/diag_translation.py --no-render      # ดูแค่ข้อความ ไม่ต้องเรนเดอร์ภาพ

ไม่ต้องใช้ API key จริง (ไม่มีการเรียก LLM ออกอินเทอร์เน็ต)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanlation.config import ScanlationConfig  # noqa: E402
from scanlation.env import apply_env, load_dotenv  # noqa: E402
from scanlation.inpainting import build_text_mask  # noqa: E402
from scanlation.schemas import BBox, BlockKind, Bubble, Page, TextBlock  # noqa: E402
from scanlation.translation.glossary import Glossary, StoryContext, load_context  # noqa: E402
from scanlation.translation.llm_translator import LLMTranslator, LLMBackend  # noqa: E402
from scanlation.typesetting import Typesetter  # noqa: E402
from scanlation.utils import image_io  # noqa: E402

# ---------------------------------------------------------------------------
# 1) หน้าทดสอบ: 4 บอลลูนที่ครอบคลุมเคสที่พบบ่อยในงานจริง
# ---------------------------------------------------------------------------
PAGE_SPEC = [
    ("bub0", BlockKind.SPEECH, [(148, 146, 366, 187), (148, 196, 392, 237)],
     "I WILL PROTECT EVERYONE I LOVE!"),
    ("bub1", BlockKind.NARRATION, [(108, 506, 421, 547), (108, 556, 387, 597)],
     "But the twin horns were nowhere to be found."),
    ("bub2", BlockKind.SFX, [(681, 214, 800, 243)], "CRACK!"),
    # บล็อกที่มีคำเฉพาะใน glossary (Mana / Mana Core) — จุดที่พบบั๊ก "ไทยปนอังกฤษ"
    ("bub3", BlockKind.SPEECH, [(120, 760, 520, 830)],
     "The Mana Core is unstable, Arthur."),
    # บล็อกที่ OCR อ่านได้แค่ขยะ (ไม่มีข้อความจริง) — เคสที่ "ไม่ถูกแปลเลย"
    ("bub4", BlockKind.SPEECH, [(600, 760, 780, 830)], "... . . . | |"),
]

TH_OK = {
    "bub0": "ข้าจะปกป้องทุกคนที่ข้าฮัก!",
    "bub1": "แต่เขาทั้งสองกลับหายสาบสูญไปอย่างไร้ร่องรอย",
    "bub2": "แครก!",
    "bub3": "แกนมานาไม่เสถียร อาเธอร์",
    "bub4": "",
}


# ---------------------------------------------------------------------------
# 2) MockBackend — เลียนแบบพฤติกรรม LLM 6 แบบที่เจอจริง
# ---------------------------------------------------------------------------
class MockBackend(LLMBackend):
    def __init__(self, scene: str):
        self.scene = scene
        self.calls = 0
        self.last_user = ""

    def _payload(self, user: str):
        ids = []
        for line in user.splitlines():
            if line.startswith("id="):
                ids.append(line.split("|")[0].replace("id=", "").strip())
        self.last_user = user

        if self.scene == "truncated_json":
            # LLM ตอบยาวเกิน max_tokens → JSON ขาดกลางคัน
            return '{"translations": [{"id": "bub0", "th": "ข้าจะปก'
        if self.scene == "empty_response":
            return ""
        if self.scene == "missing_id":
            # ตอบไม่ครบทุก id (พบบ่อยเมื่อมีหลายบรรทัด/โมเดลเล็ก)
            return {"translations": [{"id": i, "th": TH_OK[i]}
                                     for i in ids if i != "bub3"]}
        if self.scene == "kept_english_term":
            # LLM เก็บชื่อเฉพาะเป็นอังกฤษไว้ แล้ว glossary แทนไม่ครบ → ไทยปนอังกฤษ
            return {"translations": [{"id": i, "th": TH_OK[i]} for i in ids
                                     if i != "bub3"] + [{"id": "bub3", "th": "แกน Mana Core ไม่เสถียร Arthur"}]}
        if self.scene == "partial_sentence":
            # LLM แปลไม่หมดประโยค (เหลือวลีอังกฤษกลางประโยค)
            return {"translations": [{"id": i, "th": TH_OK[i]} for i in ids
                                     if i != "bub1"] + [{"id": "bub1", "th": "แต่ the twin horns กลับหายสาบสูญ"}]}
        if self.scene == "wrong_ids":
            # LLM เปลี่ยนรูปแบบ id (เช่นใส่ # หรือใช้ index)
            return {"translations": [{"id": f"#{i}", "th": TH_OK[i]} for i in ids]}
        return {"translations": [{"id": i, "th": TH_OK[i]} for i in ids]}


def make_backend(scene: str) -> LLMBackend:
    """
    MockBackend ที่เลียนแบบ LLM จริง "2 จังหวะ":
        จังหวะที่ 1 (คำขอปกติ)   -> ตอบมีปัญหา (ขาด id / เก็บอังกฤษ / id ผิด)
        จังหวะที่ 2 (รอบซ่อม)     -> ตอบถูกต้อง (เหมือน LLM จริงที่ทำตามคำกำชับได้)
    ถ้าอยากทดสอบว่าโค้ดทนแค่ไหนเมื่อ LLM "ดื้อ" ให้ตั้ง env MOCK_REPAIR_OK=0
    """
    import os
    repair_ok = os.environ.get("MOCK_REPAIR_OK", "1") != "0"

    class _B(MockBackend):
        def complete(self, system: str, user: str):
            self.calls += 1
            is_repair = "REPAIR" in user
            if scene in ("truncated_json", "empty_response"):
                # จำลอง "ตอบผิดรูปแบบทุกครั้ง" (โค้ดต้องไม่ทำให้ทั้งเรื่องล้ม)
                raise ValueError(f"MockBackend: ตอบผิดรูปแบบ (scene={scene})")
            if is_repair and repair_ok:
                ids = [ln.split("|")[0].replace("id=", "").strip()
                       for ln in user.splitlines() if ln.startswith("id=")]
                return {"translations": [{"id": i, "th": TH_OK.get(i) or "ข้อความไทยทดสอบ"}
                                         for i in ids]}
            return self._payload(user)

    return _B(scene)


# ---------------------------------------------------------------------------
# 3) helpers
# ---------------------------------------------------------------------------
def build_page(w: int, h: int) -> Page:
    page = Page(index=0, image_path="synthetic", width=w, height=h)
    for uid, kind, boxes, text in PAGE_SPEC:
        bs = [BBox(*b) for b in boxes]
        uni = bs[0]
        for b in bs[1:]:
            uni = uni.union(b)
        blocks = [TextBlock(uid=f"{uid}-l{j}", bbox=b, source_text="",
                            confidence=1.0) for j, b in enumerate(bs)]
        blocks[0].source_text = text
        page.bubbles.append(Bubble(uid=uid, bbox=uni, kind=kind, blocks=blocks))
    return page


def has_english(s: str) -> bool:
    return any(("a" <= c.lower() <= "z") for c in s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diag/trans")
    ap.add_argument("--scene", help="เลือกกรณีเดียว: ok|missing_id|kept_english_term|"
                                    "partial_sentence|wrong_ids|truncated_json|empty_response")
    ap.add_argument("--image", help="ภาพพื้นหลัง (ค่าเริ่มต้น: testdata/page-test.png)")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    cfg = apply_env(ScanlationConfig())
    cfg.inpainting.method = "median"
    still_path = Path(args.image or "testdata/page-test.png")
    still = image_io.load_image_rgb(still_path)
    h, w = still.shape[:2]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    scenes = [args.scene] if args.scene else [
        "ok", "kept_english_term", "partial_sentence",
        "missing_id", "wrong_ids", "truncated_json", "empty_response",
    ]

    report: list[str] = [
        "DIAG-TRANSLATION — จำลองพฤติกรรม LLM เพื่อหาต้นตอบอลลูน 'ไทยปนอังกฤษ'",
        f"ภาพ: {w}x{h}   backend(จำลอง)={cfg.translation.backend} "
        f"model={cfg.translation.model}",
        "",
    ]
    glossary_entries = json.loads((ROOT / "config/glossary.json").read_text(encoding="utf-8"))
    report.append(f"glossary: {glossary_entries}")
    report.append("")

    from scanlation.pipeline import ScanlationPipeline
    inp = ScanlationPipeline(cfg).inpainter
    ts = Typesetter(cfg.typesetting)

    for scene in scenes:
        page = build_page(w, h)
        base_cfg = cfg.translation
        # ปิด cache เพื่อไม่ให้ผลข้ามกันระหว่าง scenario
        base_cfg.cache_path = str(out / f"_cache_{scene}.json")
        Path(base_cfg.cache_path).unlink(missing_ok=True)

        tr = LLMTranslator(base_cfg, backend=make_backend(scene))
        # บังคับโหลด glossary เดียวกับของจริง
        tr.story = load_context(base_cfg)

        report.append(f"=== scene: {scene} ===")
        try:
            page = tr.translate_page(page)
        except Exception as exc:  # noqa: BLE001
            report.append(f"  ✗ ยกเลิกทั้งหน้า (exception): {type(exc).__name__}: {exc}")
            report.append("  → ผลคือ 'ทั้งหน้าไม่ถูกแปล' (ข้อความอังกฤษเดิมค้างทั้งหมด)")
            report.append("")
            continue

        mixed = []
        for bub in page.bubbles:
            src, th = bub.source_text, bub.translation
            flag = ""
            if not th:
                flag = "  ← ⚠ ไม่มีคำแปล (จะไม่ถูกวาดทับ)"
            elif has_english(th) and bub.kind != BlockKind.SFX:
                flag = "  ← ⚠ ยังมีอังกฤษปนในคำแปล!"
                mixed.append(bub.uid)
            report.append(f"  {bub.uid} [{bub.kind.value:9s}] ต้นฉบับ='{src}'")
            report.append(f"           คำแปล ='{th}'{flag}")
        report.append(f"  สรุป: บอลลูนที่ยังปนอังกฤษ = {mixed or 'ไม่มี'}")

        if not args.no_render:
            mask = build_text_mask(still, page.bubbles, cfg.inpainting)
            cleaned = inp.inpaint(still, mask)
            final = ts.render(cleaned, page)
            image_io.save_image(final, out / f"{scene}_final.png")
        report.append("")

    # ---- ส่วนที่ 2: ตรวจว่า "หน้าเดียวพัง" ไม่ทำให้ทั้งเรื่องไม่ถูกแปล ----
    report.append("=== pipeline.run(): ทนต่อ 'หน้าที่ล้มเหลว' ได้หรือไม่ ===")
    try:
        import shutil, tempfile
        from scanlation.pipeline import ScanlationPipeline

        tmp = Path(tempfile.mkdtemp(prefix="diag_pages_"))
        for i in (1, 2, 3):
            shutil.copy(still_path, tmp / f"page-{i:04d}.png")

        cfg2 = apply_env(ScanlationConfig())
        cfg2.inpainting.method = "median"
        # ใน sandbox ไม่มี easyocr/SAM — ใช้ tesseract+box เพื่อทดสอบเฉพาะ "ความทนทาน"
        cfg2.detection.ocr_engine = "tesseract"
        cfg2.detection.segmenter = "box"
        cfg2.detection.sam_device = "cpu"
        cfg2.inpainting.device = "cpu"
        cfg2.paths.input_dir = str(tmp)
        cfg2.paths.output_dir = str(tmp / "out")
        cfg2.translation.cache_path = str(tmp / "cache.json")

        pipe = ScanlationPipeline(cfg2)

        class _Flaky(LLMTranslator):
            """หน้าที่ index==2 ล้มเหลว (จำลอง LLM/เน็ตพัง) ส่วนหน้าอื่นแปลได้"""
            def __init__(self, *a, **k):
                super().__init__(*a, **k)

            def translate_page(self, page, context=None):
                if page.index == 2:
                    raise RuntimeError("จำลอง: LLM ตอบ JSON ขาดกลางคัน")
                for b in page.bubbles:
                    b.set_translation("ข้อความไทยทดสอบ")
                return page

        pipe._translator = _Flaky(cfg2.translation, backend=make_backend("ok"))
        import logging as _log
        from scanlation.utils.logging_utils import get_logger as _gl
        _logger = _log.getLogger("diag.scenes")
        failures: list = []
        _orig = pipe.process_page

        def _wrapped(path, index=0, do_translate=True, save=True):
            try:
                return _orig(path, index=index, do_translate=do_translate, save=save)
            except Exception as exc:  # noqa: BLE001
                failures.append((index, f"{type(exc).__name__}: {exc}"))
                raise

        pipe.process_page = _wrapped  # type: ignore[assignment]
        results = pipe.run(do_translate=True)
        report.append(f"  ประมวลผลสำเร็จ {len(results)}/3 หน้า, ล้มเหลว {len(failures)} หน้า")
        for idx, err in failures:
            report.append(f"    หน้า {idx} ล้ม: {err}")
        report.append("    ไฟล์ผลลัพธ์: "
                      f"{sorted(p.name for p in (tmp / 'out').glob('*')) if (tmp/'out').exists() else []}")
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        report.append(f"  ✗ ทดสอบไม่สำเร็จ: {type(exc).__name__}: {exc}")
    report.append("")

    report += [
        "=== สรุป: ต้นตอ + กลไกที่ซ่อม ===",
        "1. LLM ตอบไม่ครบทุก id (หรือตอบเป็นค่าว่าง / แทน id เป็น #bub0)",
        "   เดิม: llm_translator ตั้ง th = source_text  → บอลลูนนั้นกลายเป็น 'อังกฤษเต็ม ๆ'",
        "         และเพราะ enforce() ของ glossary ยังทำงานกับข้อความอังกฤษนั้นด้วย",
        "         จึงได้ 'The มานา Core is unstable, Arthur.' = ไทยปนอังกฤษตรง ๆ",
        "   ซ่อม: _verify_and_repair() ยิง 'รอบซ่อม' เฉพาะ id ที่ขาด/ว่าง + กำชับห้ามเหลืออังกฤษ",
        "         และ _apply() 'ไม่' เอา source_text มาเป็นคำแปลอีก (mark meta.untranslated)",
        "2. LLM แปลค้างวลีอังกฤษกลางประโยค ('แต่ the twin horns กลับ...')",
        "   ซ่อม: _has_leftover_english() ตรวจคำอังกฤษที่เหลือ แล้วส่งซ่อม (ผลลัพธ์ไทยล้วน)",
        "3. glossary.enforce ไล่แทนคำตามลำดับ dict → 'Mana' ถูกแทนก่อน 'Mana Core'",
        "   เหลือ 'มานา Core' = ไทยปนอังกฤษ",
        "   ซ่อม: enforce() เรียงคำ 'ยาวก่อน' (longest-first)",
        "4. ต้นฉบับเป็นขยะจาก OCR ('... . . . | |') → LLM ตอบว่าง → เดิม fallback เอาขยะมาแสดง",
        "   ซ่อม: _is_translatable() ข้ามบล็อกที่ไม่มีความหมาย (ยกเว้น SFX)",
        "5. LLM ตอบยาวเกิน max_tokens → JSON ขาดกลางคัน → exception → ทั้งหน้าไม่ถูกแปล",
        "   ซ่อม: แบ่งเป็นชุดย่อย (batch_size) + pipeline.run() จับ error 'ระดับหน้า'",
        "   → หน้าที่พังถูกข้าม แต่หน้าอื่นยังแปล+บันทึกต่อ (ไม่ทิ้งทั้งเรื่อง)",
        "",
        "กลไกที่เพิ่ม: batch_size=12, repair_passes=1 (ตั้งผ่าน env ได้)",
        "  SCANLATION_TRANSLATION_BATCH_SIZE / SCANLATION_TRANSLATION_REPAIR_PASSES",
    ]
    (out / "translation_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\n✓ รายงาน: {out}/translation_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
