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
import platform
import shutil
import sys
from pathlib import Path

# --- สีในเทอร์มินัล (ปิดอัตโนมัติถ้าไม่รองรับ) ---
_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
G = "\033[92m" if _TTY else ""
R = "\033[91m" if _TTY else ""
Y = "\033[93m" if _TTY else ""
B = "\033[1m" if _TTY else ""
X = "\033[0m" if _TTY else ""

# รายการ "สิ่งที่ต้องแก้ก่อนใช้งานได้จริง" (ใช้สรุปท้ายรายงาน)
_next_steps: list[tuple[str, str]] = []      # (ปัญหา, คำสั่ง/วิธีแก้)


# ---------------------------------------------------------------------------
# ตรวจจับ OS / ตัวจัดการแพ็กเกจ เพื่อให้คำสั่งติดตั้งตรงกับเครื่องจริง
# ---------------------------------------------------------------------------
def detect_platform() -> dict:
    system = platform.system().lower()          # linux / darwin / windows
    distro = ""
    if system == "linux":
        try:
            txt = Path("/etc/os-release").read_text(encoding="utf-8", errors="ignore")
            for line in txt.splitlines():
                if line.startswith("ID="):
                    distro = line.split("=", 1)[1].strip().strip('"').lower()
                    break
        except Exception:  # noqa: BLE001
            distro = ""
    has = lambda *cmds: any(shutil.which(c) for c in cmds)  # noqa: E731
    return {
        "system": system,
        "distro": distro,
        "conda": has("conda") or bool(os.environ.get("CONDA_DEFAULT_ENV")),
        "apt": has("apt-get", "apt") and system == "linux",
        "dnf": has("dnf") and system == "linux",
        "pacman": has("pacman") and system == "linux",
        "brew": has("brew") and system == "darwin",
        "win": system == "windows",
    }


PLAT = detect_platform()


def os_label() -> str:
    if PLAT["win"]:
        return "Windows"
    if PLAT["system"] == "darwin":
        return "macOS"
    return f"Linux ({PLAT['distro'] or 'ไม่ทราบรุ่น'})"


def cmd_libraqm() -> str:
    """คำสั่งติดตั้ง libraqm ที่ตรงกับเครื่อง + วิธีทำให้ Pillow มองเห็น"""
    if PLAT["conda"]:
        return "conda install -c conda-forge pillow   # แล้วเช็ค: python -c \"from PIL import features; print(features.check('raqm'))\""
    if PLAT["apt"]:
        return ("sudo apt-get install -y libraqm-dev && "
                "pip install --force-reinstall --no-cache-dir pillow")
    if PLAT["dnf"]:
        return ("sudo dnf install -y libraqm-devel && "
                "pip install --force-reinstall --no-cache-dir pillow")
    if PLAT["pacman"]:
        return ("sudo pacman -S --needed libraqm && "
                "pip install --force-reinstall --no-cache-dir pillow")
    if PLAT["brew"]:
        return ("brew install libraqm && "
                "pip install --force-reinstall --no-cache-dir pillow")
    if PLAT["win"]:
        return "pip install --upgrade --force-reinstall pillow   # wheel ทางการมี libraqm ในตัว"
    return "ติดตั้ง libraqm ของระบบ แล้ว pip install --force-reinstall pillow"


def cmd_tesseract() -> str:
    if PLAT["apt"]:
        return "sudo apt-get install -y tesseract-ocr"
    if PLAT["dnf"]:
        return "sudo dnf install -y tesseract"
    if PLAT["pacman"]:
        return "sudo pacman -S --needed tesseract"
    if PLAT["brew"]:
        return "brew install tesseract"
    if PLAT["win"]:
        return "ดาวน์โหลดตัวติดตั้งจาก https://github.com/UB-Mannheim/tesseract/wiki"
    return "ติดตั้งโปรแกรม tesseract ของระบบ"


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


def need(msg: str, fix: str) -> None:
    """รายการที่ 'จำเป็นต่อการรันเต็มระบบ' — เก็บไว้สรุปเป็นขั้นตอนถัดไป"""
    _next_steps.append((msg, fix))


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
    print(f"ระบบ   : {os_label()}")

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
            fail("Pillow ไม่มี libraqm — ข้อความไทยจะวางสระ/วรรณยุกต์ผิด", cmd_libraqm())
            need("libraqm (จัดสระ/วรรณยุกต์ไทย)", cmd_libraqm())
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

    # เก็บเป็นขั้นตอนถัดไป (ต้องมีถ้าจะรันเต็มระบบด้วย SAM + LaMa)
    if "segment_anything" not in sys.modules:
        need("ติดตั้ง SAM (segment-anything) + LaMa สำหรับ Detection/Inpainting",
             "pip install -r requirements-ml.txt")
    if "simple_lama_inpainting" not in sys.modules:
        need("ติดตั้ง LaMa (simple-lama-inpainting)",
             "pip install -r requirements-ml.txt")

    # ------------------------------------------------- 3.1) Tesseract (ระบบ)
    section("3.1) Tesseract OCR (ใช้เฉพาะเมื่อ ocr_engine=tesseract — ปกติไม่ต้องมี)")
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        ok("Tesseract ทำงานได้")
    except ImportError:
        warn("ไม่มี pytesseract (ข้ามได้ถ้าใช้ easyocr ที่เป็นค่าดีฟอลต์)",
             "pip install pytesseract")
    except Exception:
        warn("ไม่พบโปรแกรม tesseract ในระบบ (ข้ามได้ถ้าใช้ easyocr)",
             cmd_tesseract())

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
        need("ดาวน์โหลด checkpoint ของ SAM", "python scripts/download_weights.py --model vit_b")

    # ------------------------------------------------- 5) API key
    section("5) API key ของผู้ให้บริการ AI (สเตจแปล)")
    from scanlation.config import ScanlationConfig
    from scanlation.env import apply_env
    from scanlation.providers import get as get_provider, list_free, api_keys_for

    cfg = apply_env(ScanlationConfig())
    backend = cfg.translation.backend
    prov = get_provider(backend)
    print(f"  backend ปัจจุบัน: {backend}  (โมเดล: {cfg.translation.model})")
    if prov:
        tag = "ฟรี" if prov.free else "เสียเงิน"
        print(f"  ผู้ให้บริการ: {prov.display}  [{tag}]")
        if prov.signup_url:
            print(f"      ขอคีย์ที่: {prov.signup_url}")

    keys = api_keys_for(backend)
    have = [k for k in keys if os.environ.get(k)]
    if have:
        ok(f"พบคีย์ของ '{backend}': " + ", ".join(have))
    elif keys:
        fix = f"python -m scanlation.cli setup --provider {backend}"
        fail(f"ยังไม่พบคีย์ของ '{backend}' ({', '.join(keys)})", fix)
        need(f"ใส่ API key ของ '{backend}' (ขอคีย์ได้ที่ {prov.signup_url if prov else '-'})", fix)
    else:
        ok("ผู้ให้บริการนี้ไม่ต้องใช้คีย์ (เช่น ollama)")

    # เตือนถ้ามีคีย์ของเจ้าอื่น (อาจตั้ง backend ไม่ตรงกับคีย์ที่มี)
    from scanlation.providers import CATALOG
    others = [p.name for p in CATALOG.values()
              if p.name != backend and any(os.environ.get(k) for k in p.env_keys)]
    if others and not have:
        warn("พบคีย์ของผู้ให้บริการอื่น: " + ", ".join(others),
             "ตั้ง SCANLATION_TRANSLATION_BACKEND ให้ตรงกับคีย์ หรือเปลี่ยนคีย์")

    free_names = ", ".join(p.name for p in list_free())
    if keys and not have:
        warn(f"ผู้ให้บริการฟรีที่ใช้ได้: {free_names}",
             "รัน: python -m scanlation.cli setup --list เพื่อดูวิธีขอคีย์")

    # ------------------------------------------------- สรุปผล
    print(f"\n{B}════ สรุป ════{X}")
    print(f"  ระบบ: {os_label()}  |  python {sys.version.split()[0]}")
    print(f"  ผ่าน {ok_count} | เตือน {warn_count} | ไม่ผ่าน {fail_count}")

    if _next_steps:
        print(f"\n{B}ขั้นตอนถัดไป (เรียงตามลำดับที่แนะนำ){X}")
        for i, (what, cmd) in enumerate(_next_steps, 1):
            print(f"  {i}. {what}")
            if cmd:
                print(f"       {G}{cmd}{X}")

    if fail_count == 0:
        print(f"\n  {G}สภาพแวดล้อมพร้อมใช้งาน 🎉{X}")
        return 0
    print(f"\n  {R}ยังมี {fail_count} รายการที่ต้องแก้ (ดู '→' ด้านบน){X}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
