#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/check_env.py
====================
ตรวจว่า "สภาพแวดล้อม" ของโปรเจกต์พร้อมใช้งานหรือยัง (รันได้ทุก OS)

    python scripts/check_env.py
    python scripts/check_env.py --strict    # ให้ dependency ที่ขาดเป็น "ไม่ผ่าน" ทั้งหมด

รายงานจะบอกทีละรายการว่าอะไร ✓ พร้อม / ✗ ยังขาด พร้อมวิธีแก้สั้น ๆ
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

# --- สีในเทอร์มินัล (ปิดอัตโนมัติถ้าไม่รองรับ) ---
_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
G = "\033[92m" if _TTY else ""
R = "\033[91m" if _TTY else ""
Y = "\033[93m" if _TTY else ""
B = "\033[1m" if _TTY else ""
X = "\033[0m" if _TTY else ""

ok_count = 0
fail_count = 0
warn_count = 0


def ok(msg: str) -> None:
    global ok_count
    ok_count += 1
    print(f"  {G}✓{X} {msg}")


def fail(msg: str, fix: str = "") -> None:
    global fail_count
    fail_count += 1
    print(f"  {R}✗{X} {msg}")
    if fix:
        print(f"      {Y}→ {fix}{X}")


def warn(msg: str, fix: str = "") -> None:
    global warn_count
    warn_count += 1
    print(f"  {Y}!{X} {msg}")
    if fix:
        print(f"      {Y}→ {fix}{X}")


def section(title: str) -> None:
    print(f"\n{B}{title}{X}")


def check_import(module: str, label: str, required: bool = True, fix: str = "") -> bool:
    try:
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "")
        ok(f"{label} {'(' + str(ver) + ')' if ver else ''}")
        return True
    except Exception as exc:  # noqa: BLE001
        (fail if required else warn)(f"{label} — ไม่พร้อม ({type(exc).__name__})", fix)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="ตรวจสภาพแวดล้อมของ Scanlation Engine")
    ap.add_argument("--strict", action="store_true",
                    help="ให้ dependency ที่ขาดถือเป็น 'ไม่ผ่าน' (ค่าเริ่มต้นเป็นแค่คำเตือน)")
    args = ap.parse_args()
    opt = not args.strict

    # ให้ import scanlation ได้เมื่อรันจาก root ของโปรเจกต์
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # โหลด .env (ถ้ามี) ก่อนตรวจทุกอย่าง
    try:
        from scanlation.env import load_dotenv
        load_dotenv(root / ".env")
    except Exception:  # noqa: BLE001
        pass

    print(f"{B}════ Scanlation Engine — ตรวจสภาพแวดล้อม ════{X}")
    print(f"python : {sys.version.split()[0]}  ({sys.executable})")

    # ------------------------------------------------------- 0) .env / env vars
    section("0) ไฟล์ .env และ environment variable")
    from scanlation.env import env_report, KNOWN_AI_KEYS, ENV_MAP
    repo_root = Path(__file__).resolve().parent.parent
    env_file = repo_root / ".env"
    example = repo_root / ".env.example"
    if env_file.exists():
        ok(f"พบไฟล์ .env ({env_file.name}) — ระบบจะโหลดอัตโนมัติ")
    else:
        hint = ".env.example มีให้แล้ว" if example.exists() else "สร้างจากตัวอย่าง"
        warn("ยังไม่มีไฟล์ .env",
             f"คัดลอกตัวอย่างแล้วแก้ค่า:  cp .env.example .env   ({hint})")
    try:
        import dotenv  # noqa: F401
        ok("มีไลบรารี python-dotenv (โหลด .env ได้ครบ syntax)")
    except ImportError:
        warn("ไม่มี python-dotenv — จะใช้ตัว parse ในตัวแทน (ก็ใช้ได้)",
             "pip install python-dotenv")

    rep = env_report()
    if rep["app_vars_set"]:
        ok(f"มี SCANLATION_* ตั้งไว้ {len(rep['app_vars_set'])} ตัว: "
           + ", ".join(rep["app_vars_set"][:6])
           + (" …" if len(rep["app_vars_set"]) > 6 else ""))
    else:
        warn(f"ยังไม่ได้ตั้ง SCANLATION_* ใด ๆ (ใช้ค่าดีฟอลต์จาก config — มี {len(ENV_MAP)} ตัวที่ตั้งได้)",
             "ดูรายการทั้งหมดใน .env.example")
    if rep["ai_keys_set"]:
        ok("พบ API key ที่ตั้งไว้: " + ", ".join(rep["ai_keys_set"]))
    else:
        warn("ยังไม่พบ API key ของผู้ให้บริการ AI ใด ๆ",
             "ใส่ใน .env (ดูรายการครบใน .env.example) — "
             f"รองรับ {len(KNOWN_AI_KEYS)} คีย์")

    # ---------------------------------------------------------------- 1) Python
    section("1) Python")
    if sys.version_info >= (3, 9):
        ok(f"เวอร์ชัน Python {sys.version_info.major}.{sys.version_info.minor} ใช้ได้ "
           f"(แนะนำ 3.11)")
    else:
        fail(f"Python {sys.version_info.major}.{sys.version_info.minor} เก่าเกินไป",
             "ติดตั้ง Python 3.11")

    # ------------------------------------------------- 2) dependency พื้นฐาน
    section("2) Dependency พื้นฐาน (บังคับ)")
    base = [
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("requests", "requests"),
    ]
    for mod, label in base:
        check_import(mod, label, required=True, fix="pip install -r requirements.txt")

    # pythainlp (ตัดคำไทย) — ถ้าขาดจะ fallback เป็นตัดทีละตัวอักษร (คุณภาพลดลง)
    check_import("pythainlp", "pythainlp (ตัดคำไทย)", required=True,
                 fix="pip install -r requirements.txt")

    # ------------------------------------------------- 2.1) raqm (สำคัญมาก)
    section("2.1) การจัดวางสระ/วรรณยุกต์ไทย (libraqm)")
    try:
        from PIL import features
        if features.check("raqm"):
            ok("Pillow มี libraqm — จัดสระ/วรรณยุกต์ไทยได้ถูกต้อง")
        else:
            fail("Pillow ไม่มี libraqm — ข้อความไทยจะวางสระ/วรรณยุกต์ผิด",
                 "Linux: sudo apt install libraqm-dev ; Conda: conda install -c conda-forge pillow")
    except Exception as exc:  # noqa: BLE001
        fail(f"ตรวจ raqm ไม่ได้ ({exc})", "อัปเกรด Pillow: pip install -U pillow")

    # ------------------------------------------------- 2.2) WebP support
    section("2.2) รองรับ WebP (สเตจดาวน์โหลด)")
    try:
        from PIL import features
        if features.check("webp"):
            ok("Pillow บันทึก/อ่าน WebP ได้")
        else:
            fail("Pillow ไม่รองรับ WebP", "ติดตั้ง Pillow ใหม่จาก wheel ทางการ: pip install -U pillow")
    except Exception as exc:  # noqa: BLE001
        fail(f"ตรวจ WebP ไม่ได้ ({exc})")

    # ------------------------------------------------- 3) dependency ML (ทางเลือก)
    section("3) Dependency โมเดล ML (สำหรับ pipeline เต็มระบบ)")
    check_import("torch", "PyTorch (torch)", required=False, fix=
                 "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    torch_ok = "torch" in sys.modules
    if torch_ok:
        try:
            import torch  # noqa: F401
            if torch.cuda.is_available():
                ok(f"CUDA ใช้ได้ — GPU: {torch.cuda.get_device_name(0)}")
            else:
                warn("พบ torch แต่ใช้ CPU (ไม่มี CUDA) — จะช้ามาก",
                     "ถ้ามี GPU: ติดตั้ง torch เวอร์ชัน CUDA ให้ตรงเครื่อง")
        except Exception as exc:  # noqa: BLE001
            warn(f"ตรวจ CUDA ไม่ได้ ({exc})")

    check_import("segment_anything", "segment-anything (SAM)", required=False,
                 fix="pip install git+https://github.com/facebookresearch/segment-anything.git")
    check_import("easyocr", "easyocr (OCR)", required=False,
                 fix="pip install -r requirements.txt")
    check_import("simple_lama_inpainting", "simple-lama-inpainting (LaMa)", required=False,
                 fix="pip install -r requirements-ml.txt")
    check_import("ultralytics", "ultralytics (FastSAM)", required=False,
                 fix="pip install -r requirements-ml.txt")
    check_import("anthropic", "anthropic SDK (Claude)", required=False,
                 fix="pip install -r requirements.txt")
    check_import("openai", "openai SDK (GPT)", required=False,
                 fix="pip install -r requirements.txt")

    # ------------------------------------------------- 3.1) Tesseract (ระบบ)
    section("3.1) Tesseract OCR (ถ้าเลือก ocr_engine=tesseract)")
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        ok("Tesseract ทำงานได้")
    except ImportError:
        warn("ไม่มี pytesseract", "pip install pytesseract")
    except Exception:
        warn("ไม่พบโปรแกรม tesseract ในระบบ",
             "Ubuntu: sudo apt install tesseract-ocr ; macOS: brew install tesseract")

    # ------------------------------------------------- 4) ไฟล์ในโปรเจกต์
    section("4) ไฟล์ / โฟลเดอร์ที่โปรเจกต์ต้องใช้")
    root = Path(__file__).resolve().parent.parent
    for rel in ["fonts/Sarabun-Regular.ttf", "fonts/Sarabun-Bold.ttf",
                "config/config.json", "config/characters.json", "config/glossary.json"]:
        p = root / rel
        if p.exists():
            ok(rel)
        else:
            fail(rel, "ไฟล์หาย — clone repo ใหม่")

    # ------------------------------------------------- 4.1) SAM weights
    section("4.1) Checkpoint ของโมเดล (SAM)")
    weights = list((root / "weights").glob("sam_*.pth")) if (root / "weights").exists() else []
    if weights:
        for w in weights:
            size_mb = w.stat().st_size / 1e6
            ok(f"weights/{w.name}  ({size_mb:.0f} MB)")
    else:
        warn("ไม่พบไฟล์ weights/sam_*.pth (จำเป็นเมื่อใช้ SAMSegmenter)",
             "python scripts/download_weights.py --model vit_b")

    # ------------------------------------------------- 5) API key
    section("5) API key ของ LLM (สเตจแปล)")
    a = bool(os.environ.get("ANTHROPIC_API_KEY"))
    o = bool(os.environ.get("OPENAI_API_KEY"))
    if a:
        ok("ANTHROPIC_API_KEY ตั้งแล้ว (Claude)")
    else:
        warn("ยังไม่ตั้ง ANTHROPIC_API_KEY", "ดูวิธีตั้งค่าใน SETUP.md หัวข้อ 'ตั้งค่า API key'")
    if o:
        ok("OPENAI_API_KEY ตั้งแล้ว (GPT)")
    else:
        warn("ยังไม่ตั้ง OPENAI_API_KEY")
    if not (a or o):
        warn("ไม่มี API key ใด ๆ — สเตจแปลจะใช้ไม่ได้ (แต่ demo/ดาวน์โหลดยังใช้ได้)")

    # ------------------------------------------------- สรุปผล
    print(f"\n{B}════ สรุป ════{X}")
    print(f"  ผ่าน {ok_count} | เตือน {warn_count} | ไม่ผ่าน {fail_count}")
    if fail_count == 0:
        print(f"  {G}สภาพแวดล้อมพื้นฐานพร้อมใช้งาน 🎉{X}")
        return 0
    print(f"  {R}ยังมีรายการที่ต้องแก้ (ดูข้อ '→' ด้านบน){X}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
