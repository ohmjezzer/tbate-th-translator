# -*- coding: utf-8 -*-
"""
cli.py
======
Command-line interface ของระบบ

ตัวอย่างการใช้:
    python -m scanlation.cli run            # แปลทั้งเรื่อง (Detection→Inpaint→Translate→Typeset)
    python -m scanlation.cli run 4 23 26    # เฉพาะบางหน้า
    python -m scanlation.cli detect 4       # ทดสอบสเตจตรวจจับ + SAVE mask ไว้ดู
    python -m scanlation.cli init-config    # สร้างไฟล์ config ตัวอย่าง
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ScanlationConfig
from .pipeline import ScanlationPipeline
from .translation import StoryContext, LLMTranslator
from .translation.prompts import build_system_prompt
from .utils.logging_utils import get_logger

log = get_logger("scanlation.cli")


def _load_config(path: str | None) -> ScanlationConfig:
    if path and Path(path).exists():
        return ScanlationConfig.load(path)
    return ScanlationConfig()


def cmd_run(args) -> int:
    cfg = _load_config(args.config)
    pipeline = ScanlationPipeline(cfg)
    pages = [int(x) for x in args.pages] if args.pages else None
    results = pipeline.run(pages=pages, do_translate=not args.no_translate)
    log.info("เสร็จสิ้น %d หน้า", len(results))
    return 0


def cmd_detect(args) -> int:
    """ทดสอบเฉพาะสเตจ Detection & Masking (บันทึก mask ไว้ตรวจด้วยตา)"""
    import numpy as np

    from .detection import DetectionPipeline
    from .detection.ocr_detector import EasyOCRDetector
    from .detection.sam_segmenter import SAMSegmenter
    from .utils import image_io

    cfg = _load_config(args.config)
    d = cfg.detection
    det = DetectionPipeline(EasyOCRDetector(d), SAMSegmenter(d), d)
    pipe = ScanlationPipeline(cfg, detection=det)

    for index, path in pipe.discover_pages():
        if args.pages and index not in {int(x) for x in args.pages}:
            continue
        image = image_io.load_image_rgb(path)
        page = det.detect_page(image, index=index, image_path=str(path))
        print(f"\n=== หน้า {index} : {path.name} ===")
        for bub in page.bubbles:
            print(f"  [{bub.kind.value:9s}] box={bub.bbox.to_int()} "
                  f"'{bub.source_text[:60]}'")
            # วาด mask ทับเพื่อดูผล (บันทึกเป็นไฟล์)
            if bub.mask is not None:
                overlay = image.copy()
                overlay[bub.mask] = (overlay[bub.mask] * 0.4 + np.array([255, 0, 0]) * 0.6)
                image_io.save_image(overlay, pipe._work_path(f"mask_preview_p{index:04d}.png"))
    return 0


def cmd_show_prompt(args) -> int:
    """พิมพ์ system prompt ที่จะส่งให้ LLM (ไว้ตรวจ/ปรับบริบท)"""
    cfg = _load_config(args.config)
    story = StoryContext.from_json(cfg.translation.characters_path, cfg.translation.glossary_path)
    from .translation.base import TranslationContext

    print(build_system_prompt(TranslationContext(story=story)))
    return 0


def cmd_init_config(args) -> int:
    """สร้างไฟล์ config + glossary/characters ตัวอย่าง"""
    cfg = ScanlationConfig()
    out = Path(args.dir or "config")
    out.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out / "config.json")
    (out / "glossary.json").write_text(
        '{\n  "Realmheart": "เรียลม์ฮาร์ต",\n  "Mana": "มานา"\n}\n', encoding="utf-8"
    )
    (out / "characters.json").write_text(
        '{\n'
        '  "synopsis": "เรื่องย่อ...",\n'
        '  "style_guide": "ใช้ภาษาไทยธรรมชาติ",\n'
        '  "characters": [\n'
        '    {"name": "Arthur", "aliases": ["อาเธอร์"], "description": "ตัวเอก", '
        '"pronouns": "ฉัน", "address_others": "เธอ", "speech_style": "จริงจัง สุภาพ"}\n'
        '  ]\n}\n', encoding="utf-8"
    )
    print(f"สร้างไฟล์ config ตัวอย่างที่ {out}/ เรียบร้อย")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scanlation", description="ระบบแปลมังงะอัตโนมัติ")
    p.add_argument("-c", "--config", help="path ของ config.json/yaml")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="รัน pipeline เต็มรูปแบบ")
    pr.add_argument("pages", nargs="*", help="เลขหน้าที่ต้องการ (ว่าง=ทุกหน้า)")
    pr.add_argument("--no-translate", action="store_true", help="ข้ามสเตจแปล (ทดสอบ inpaint+typeset)")
    pr.set_defaults(func=cmd_run)

    pd = sub.add_parser("detect", help="ทดสอบสเตจตรวจจับ")
    pd.add_argument("pages", nargs="*")
    pd.set_defaults(func=cmd_detect)

    pp = sub.add_parser("show-prompt", help="พิมพ์ system prompt สำหรับ LLM")
    pp.set_defaults(func=cmd_show_prompt)

    pi = sub.add_parser("init-config", help="สร้าง config ตัวอย่าง")
    pi.add_argument("--dir", default="config")
    pi.set_defaults(func=cmd_init_config)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
