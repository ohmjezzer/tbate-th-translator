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
from .env import apply_env, load_dotenv
from .pipeline import ScanlationPipeline
from .translation import StoryContext, LLMTranslator
from .translation.prompts import build_system_prompt
from .utils.logging_utils import get_logger

log = get_logger("scanlation.cli")


def _load_config(path: str | None) -> ScanlationConfig:
    """
    สร้าง config โดยโหลด .env ก่อน แล้วให้ env ทับค่าจากไฟล์ config
    (ลำดับความสำคัญ: env ในเชลล์ > .env > ไฟล์ config > ค่าดีฟอลต์)
    """
    load_dotenv()                       # อ่าน .env อัตโนมัติถ้ามี
    if path and Path(path).exists():
        return ScanlationConfig.load(path)   # load() จะ apply env ให้แล้ว
    if path:
        log.warning("ไม่พบไฟล์ config: %s — ใช้ค่าดีฟอลต์ + env", path)
    return apply_env(ScanlationConfig())


def cmd_run(args) -> int:
    cfg = _load_config(args.config)
    # ตรวจล่วงหน้า: ถ้าจะแปลแต่ยังไม่มีคีย์ ให้ข้อความแนะนำเป็นขั้นตอน (ไม่ให้ล้มกลางทาง)
    if not args.no_translate and not cfg.api_key():
        from .providers import missing_key_help

        print(missing_key_help(cfg.translation.backend,
                               command="python -m scanlation.cli run"), file=sys.stderr)
        return 2
    pipeline = ScanlationPipeline(cfg)
    pages = [int(x) for x in args.pages] if args.pages else None
    results = pipeline.run(pages=pages, do_translate=not args.no_translate)
    log.info("เสร็จสิ้น %d หน้า", len(results))
    return 0


def cmd_download(args) -> int:
    """
    สเตจ 0: ดาวน์โหลดรูปจากเว็บ → บังคับแปลงเป็น WebP → วางใน input_dir (orig/)

        python -m scanlation.cli download <chapter_url>
        python -m scanlation.cli download <chapter_url> --source vortexscans --out orig
        python -m scanlation.cli download <chapter_url> --run        # ดาวน์โหลดแล้วแปลต่อทันที
    """
    cfg = _load_config(args.config)
    # CLI override
    if args.source:
        cfg.download.source = args.source
    if args.out:
        cfg.download.out_dir = args.out
    if args.quality:
        cfg.download.webp_quality = args.quality
    if args.prefix:
        cfg.download.prefix = args.prefix
    if args.workers:
        cfg.download.max_workers = args.workers
    if getattr(args, "limit", 0):
        cfg.download.limit = args.limit
    if getattr(args, "urls_file", None):
        cfg.download.image_urls_file = args.urls_file

    pipeline = ScanlationPipeline(cfg)

    if getattr(args, "run", False):
        # ดาวน์โหลดแล้วแปลต่อทันที
        results = pipeline.download_and_run(args.chapter_url)
        log.info("เสร็จสิ้น: ดาวน์โหลด + แปล %d หน้า", len(results))
    else:
        report = pipeline.download(args.chapter_url)
        print(report.summary())
        if report.errors:
            print("หน้าที่ล้มเหลว:")
            for url, err in list(report.errors.items())[:10]:
                print(f"  - {url}\n    {err}")
    return 0


def cmd_list_sources(args) -> int:
    """แสดงรายการ source ที่ลงทะเบียนไว้"""
    from . import downloader  # noqa: F401  (import เพื่อให้ @register ทำงาน)
    from .utils.registry import available
    print("Download sources ที่ใช้ได้:", ", ".join(available("source")))
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


# ===========================================================================
# setup — ตั้งค่าครั้งแรกด้วยผู้ให้บริการ "ฟรี" (ไม่ต้องมีโมเดล/บัตรเครดิต)
# ===========================================================================
def _print_catalog(free_only: bool = False) -> None:
    from .providers import list_all, list_free

    providers = list_free() if free_only else list_all()
    print("\n=== ผู้ให้บริการที่รองรับ ===")
    print(f"{'ชื่อ':<12} {'ฟรี':<5} {'โมเดลดีฟอลต์':<44} ขอคีย์ที่")
    print("-" * 96)
    for p in providers:
        print(f"{p.name:<12} {'✓' if p.free else '✗':<5} {p.default_model:<44} {p.signup_url}")
    print("\nหมายเหตุ: ผู้ให้บริการที่ไม่ใช่ Anthropic ใช้ API แบบ OpenAI-compatible "
          "ระบบตั้ง base_url ให้อัตโนมัติ")


def cmd_setup(args) -> int:
    """
    ตั้งค่าครั้งแรก (สร้างไฟล์ .env) ด้วยผู้ให้บริการฟรี

        python -m scanlation.cli setup --list                 # ดูรายชื่อผู้ให้บริการ
        python -m scanlation.cli setup --provider gemini --key AIza... # เขียน .env
        python -m scanlation.cli setup --free                 # ใช้ผู้ให้บริการฟรีที่แนะนำ (มีช่องให้เติมคีย์)
    """
    from .providers import get, list_free

    if args.list:
        _print_catalog(free_only=not args.all)
        return 0

    target = args.path or ".env"
    path = Path(target)

    # เลือกผู้ให้บริการ: ถ้าระบุ --provider ใช้ตัวนั้น, ถ้า --free ใช้ตัวฟรีที่แนะนำตัวแรก
    if args.provider:
        prov = get(args.provider)
        if prov is None:
            print(f"ไม่รู้จักผู้ให้บริการ '{args.provider}' — ดูรายชื่อ: "
                  f"python -m scanlation.cli setup --list")
            return 2
    else:
        prov = list_free()[0]      # gemini

    if path.exists() and not args.force:
        print(f"ไฟล์ {path} มีอยู่แล้ว — ใช้ --force ถ้าต้องการเขียนทับ")
        return 1

    key = args.key or ""
    key_line = f"{prov.primary_env()}={key}" if prov.primary_env() else (
        "# (ผู้ให้บริการนี้ไม่ต้องใช้คีย์ — รันในเครื่อง)")

    content = f"""# =============================================================
#  .env — สร้างโดย `python -m scanlation.cli setup`
#  ผู้ให้บริการ: {prov.display}{"  (ระดับฟรี)" if prov.free else ""}
#  ขอคีย์ (ถ้ามี) ที่: {prov.signup_url}
# =============================================================

# --- AI (สเตจแปล) ---
{key_line}
SCANLATION_TRANSLATION_BACKEND={prov.name}
SCANLATION_TRANSLATION_MODEL={prov.default_model}

# --- อุปกรณ์ (cpu | cuda | mps) ---
SCANLATION_PROFILE=cpu

#  ต้องการปรับค่าอื่น ๆ ดูรายการทั้งหมดได้ในไฟล์ .env.example
"""
    path.write_text(content, encoding="utf-8")
    print(f"✓ เขียนไฟล์ {path} เรียบร้อย (ผู้ให้บริการ: {prov.display})")
    if prov.env_keys and not key:
        print(f"  → ต่อไป: ขอคีย์ฟรีได้ที่ {prov.signup_url}")
        print(f"          แล้วใส่ในไฟล์ {path} บรรทัด: {prov.primary_env()}=<คีย์ของคุณ>")
    print(f"  → ตรวจสอบ: python scripts/check_env.py")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scanlation", description="ระบบแปลมังงะอัตโนมัติ")
    p.add_argument("-c", "--config", help="path ของ config.json/yaml")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="รัน pipeline เต็มรูปแบบ")
    pr.add_argument("pages", nargs="*", help="เลขหน้าที่ต้องการ (ว่าง=ทุกหน้า)")
    pr.add_argument("--no-translate", action="store_true", help="ข้ามสเตจแปล (ทดสอบ inpaint+typeset)")
    pr.set_defaults(func=cmd_run)

    pdl = sub.add_parser("download", help="ดาวน์โหลดรูปจากเว็บ → บังคับ WebP → orig/")
    pdl.add_argument("chapter_url", help="URL ของ chapter (เช่น https://vortexscans.org/series/<slug>/chapter-255)")
    pdl.add_argument("--source", help="ชื่อ source (vortexscans/generic-json/generic-html)")
    pdl.add_argument("--out", help="โฟลเดอร์ปลายทาง (ดีฟอลต์ = paths.input_dir)")
    pdl.add_argument("--prefix", help="คำนำหน้าชื่อไฟล์ (ดีฟอลต์ page)")
    pdl.add_argument("--quality", type=int, help="คุณภาพ WebP 0-100")
    pdl.add_argument("--workers", type=int, help="จำนวนเธรดดาวน์โหลด")
    pdl.add_argument("--limit", type=int, help="ดาวน์โหลดแค่ N หน้าแรก (0=ทั้งหมด)")
    pdl.add_argument("--urls-file", dest="urls_file", help="ไฟล์ JSON รายการ URL (generic-json)")
    pdl.add_argument("--run", action="store_true", help="ดาวน์โหลดเสร็จแล้วแปลต่อทันที")
    pdl.set_defaults(func=cmd_download)

    ps = sub.add_parser("sources", help="แสดงรายการ download source ที่ใช้ได้")
    ps.set_defaults(func=cmd_list_sources)

    pd = sub.add_parser("detect", help="ทดสอบสเตจตรวจจับ")
    pd.add_argument("pages", nargs="*")
    pd.set_defaults(func=cmd_detect)

    pp = sub.add_parser("show-prompt", help="พิมพ์ system prompt สำหรับ LLM")
    pp.set_defaults(func=cmd_show_prompt)

    pi = sub.add_parser("init-config", help="สร้าง config ตัวอย่าง")
    pi.add_argument("--dir", default="config")
    pi.set_defaults(func=cmd_init_config)

    pst = sub.add_parser("setup", help="ตั้งค่าครั้งแรกด้วยผู้ให้บริการ AI ฟรี (สร้าง .env)")
    pst.add_argument("--list", action="store_true", help="แสดงรายชื่อผู้ให้บริการ")
    pst.add_argument("--all", action="store_true", help="แสดงทั้งหมด (รวมตัวเสียเงิน)")
    pst.add_argument("--provider", help="ชื่อผู้ให้บริการ เช่น gemini/groq/openrouter/anthropic/openai")
    pst.add_argument("--key", help="API key (จะถูกเขียนลง .env)")
    pst.add_argument("--free", action="store_true", help="ใช้ผู้ให้บริการฟรีที่แนะนำ")
    pst.add_argument("--path", default=".env", help="ไฟล์ปลายทาง (ดีฟอลต์ .env)")
    pst.add_argument("--force", action="store_true", help="เขียนทับไฟล์เดิม")
    pst.set_defaults(func=cmd_setup)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
