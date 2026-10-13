# -*- coding: utf-8 -*-
"""
env.py
======
ระบบโหลดค่าตั้งต้นจาก environment variable / ไฟล์ `.env`

แนวคิด (12-factor app):
    - ค่า config ทุกตัวสามารถ override ได้ด้วย environment variable
    - API key ของผู้ให้บริการ LLM ใช้ชื่อมาตรฐานที่ SDK อ่านเองอยู่แล้ว
      (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...)
    - ค่าตั้งของแอปใช้คำนำหน้า SCANLATION_ เพื่อกันชนกับตัวแปรอื่นในเครื่อง

การใช้งาน:
    from scanlation.env import load_dotenv, apply_env
    load_dotenv()                 # อ่านไฟล์ .env (ถ้ามี) เข้า os.environ
    cfg = ScanlationConfig()
    apply_env(cfg)                # ให้ env ทับค่าดีฟอลต์/ค่าจากไฟล์ config

ลำดับความสำคัญ (priority):  env จริงในเชลล์  >  ไฟล์ .env  >  config.json  >  ค่าดีฟอลต์
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# ตารางจับคู่: ชื่อ environment variable -> (section, field, ชนิดข้อมูล)
# ---------------------------------------------------------------------------
Entry = Tuple[str, str, Callable[[str], Any]]


def _bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y")


def _int(v: str) -> int:
    return int(float(v))


def _float(v: str) -> float:
    return float(v)


def _str(v: str) -> str:
    return str(v)


def _csv_tuple(v: str) -> tuple:
    return tuple(x.strip() for x in str(v).split(",") if x.strip() != "")


# (env_name, section, field, type)
ENV_MAP: Dict[str, Tuple[str, str, Callable[[str], Any]]] = {
    # ---------------- paths ----------------
    "SCANLATION_INPUT_DIR":   ("paths", "input_dir", _str),
    "SCANLATION_OUTPUT_DIR":  ("paths", "output_dir", _str),
    "SCANLATION_WORK_DIR":    ("paths", "work_dir", _str),
    "SCANLATION_FONTS_DIR":   ("paths", "fonts_dir", _str),
    "SCANLATION_CONFIG_DIR":  ("paths", "config_dir", _str),

    # ---------------- download ----------------
    "SCANLATION_DOWNLOAD_SOURCE":      ("download", "source", _str),
    "SCANLATION_DOWNLOAD_OUT_DIR":     ("download", "out_dir", _str),
    "SCANLATION_DOWNLOAD_PREFIX":      ("download", "prefix", _str),
    "SCANLATION_DOWNLOAD_START_INDEX": ("download", "start_index", _int),
    "SCANLATION_WEBP_QUALITY":         ("download", "webp_quality", _int),
    "SCANLATION_WEBP_METHOD":          ("download", "webp_method", _int),
    "SCANLATION_DOWNLOAD_MAX_WIDTH":   ("download", "max_width", _int),
    "SCANLATION_FORCE_WEBP":           ("download", "force_webp", _bool),
    "SCANLATION_DOWNLOAD_OVERWRITE":   ("download", "overwrite", _bool),
    "SCANLATION_DOWNLOAD_LIMIT":       ("download", "limit", _int),
    "SCANLATION_DOWNLOAD_WORKERS":     ("download", "max_workers", _int),
    "SCANLATION_DOWNLOAD_RETRIES":     ("download", "retries", _int),
    "SCANLATION_DOWNLOAD_TIMEOUT":     ("download", "timeout", _float),
    "SCANLATION_DOWNLOAD_CHUNK_SIZE":  ("download", "chunk_size", _int),
    "SCANLATION_DOWNLOAD_USER_AGENT":  ("download", "user_agent", _str),
    "SCANLATION_DOWNLOAD_REFERER":     ("download", "referer", _str),
    "SCANLATION_MANIFEST_PATH":        ("download", "manifest_path", _str),
    "SCANLATION_IMAGE_URLS_FILE":      ("download", "image_urls_file", _str),
    "SCANLATION_SOURCE_REGEX":         ("download", "source_regex", _str),

    # ---------------- detection ----------------
    "SCANLATION_SEGMENTER":          ("detection", "segmenter", _str),
    "SCANLATION_AUTO_FALLBACK":      ("detection", "auto_fallback", _bool),
    "SCANLATION_OCR_ENGINE":         ("detection", "ocr_engine", _str),
    "SCANLATION_OCR_LANGS":          ("detection", "ocr_langs", _csv_tuple),
    "SCANLATION_OCR_MIN_CONFIDENCE": ("detection", "ocr_min_confidence", _float),
    "SCANLATION_MERGE_GAP":          ("detection", "merge_gap", _int),
    "SCANLATION_SAM_MODEL_TYPE":     ("detection", "sam_model_type", _str),
    "SCANLATION_SAM_CHECKPOINT":     ("detection", "sam_checkpoint", _str),
    "SCANLATION_SAM_DEVICE":         ("detection", "sam_device", _str),
    "SCANLATION_SAM_EXPAND":         ("detection", "sam_expand", _int),
    "SCANLATION_SAM_MASK_FILL_RATIO": ("detection", "sam_mask_fill_ratio", _float),
    "SCANLATION_SAM_BUBBLE_WHITENESS": ("detection", "sam_bubble_whiteness", _float),
    "SCANLATION_USE_AUTO_MASK":      ("detection", "use_auto_mask", _bool),

    # ---------------- inpainting ----------------
    "SCANLATION_INPAINT_METHOD":  ("inpainting", "method", _str),
    "SCANLATION_INPAINT_DEVICE":  ("inpainting", "device", _str),
    "SCANLATION_INPAINT_AUTO_FALLBACK": ("inpainting", "auto_fallback", _bool),
    "SCANLATION_LAMA_CHECKPOINT": ("inpainting", "lama_checkpoint", _str),
    "SCANLATION_LAMA_AUTO_DOWNLOAD": ("inpainting", "lama_auto_download", _bool),
    "SCANLATION_LAMA_REGION_PAD": ("inpainting", "lama_region_pad", _int),
    "SCANLATION_MASK_DILATE":     ("inpainting", "mask_dilate", _int),
    "SCANLATION_TEXT_DELTA":      ("inpainting", "text_delta", _int),
    "SCANLATION_OPENCV_RADIUS":   ("inpainting", "opencv_radius", _int),
    "SCANLATION_OPENCV_METHOD":   ("inpainting", "opencv_method", _str),
    "SCANLATION_PAD_TO_MULTIPLE": ("inpainting", "pad_to_multiple", _int),

    # ---------------- translation ----------------
    "SCANLATION_TRANSLATION_BACKEND":    ("translation", "backend", _str),
    "SCANLATION_TRANSLATION_MODEL":      ("translation", "model", _str),
    "SCANLATION_TRANSLATION_TEMPERATURE": ("translation", "temperature", _float),
    "SCANLATION_TRANSLATION_MAX_TOKENS": ("translation", "max_tokens", _int),
    "SCANLATION_TARGET_LANG":            ("translation", "target_lang", _str),
    "SCANLATION_GLOSSARY_PATH":          ("translation", "glossary_path", _str),
    "SCANLATION_CHARACTERS_PATH":        ("translation", "characters_path", _str),
    "SCANLATION_TRANSLATION_CACHE":      ("translation", "cache_path", _str),
    "SCANLATION_TRANSLATION_MAX_RETRIES": ("translation", "max_retries", _int),
    "SCANLATION_TRANSLATION_TIMEOUT":    ("translation", "timeout", _float),
    "SCANLATION_TRANSLATION_BASE_URL":   ("translation", "base_url", _str),

    # ---------------- typesetting ----------------
    "SCANLATION_FONT_REGULAR":          ("typesetting", "regular_font", _str),
    "SCANLATION_FONT_BOLD":             ("typesetting", "bold_font", _str),
    "SCANLATION_FONT_SFX":              ("typesetting", "sfx_font", _str),
    "SCANLATION_MIN_FONT_SIZE":         ("typesetting", "min_font_size", _int),
    "SCANLATION_MAX_FONT_SIZE":         ("typesetting", "max_font_size", _int),
    "SCANLATION_LINE_SPACING":          ("typesetting", "line_spacing", _float),
    "SCANLATION_PADDING_RATIO":         ("typesetting", "padding_ratio", _float),
    "SCANLATION_SFX_STROKE_RATIO":      ("typesetting", "sfx_stroke_ratio", _float),
    "SCANLATION_AUTO_CONTRAST":         ("typesetting", "auto_contrast", _bool),
    "SCANLATION_FIT_ATTEMPTS":          ("typesetting", "fit_attempts", _int),
}

# ---------------------------------------------------------------------------
# API key ของผู้ให้บริการ AI — ดึงจากแคตตาล็อกกลาง (scanlation/providers.py)
# เพื่อไม่ให้จำนวน/ชื่อคีย์กระจายอยู่หลายที่ (single source of truth)
# ---------------------------------------------------------------------------
from . import providers as _providers  # noqa: E402  (import หลัง stdlib)

PROVIDER_API_KEYS: Dict[str, List[str]] = {
    name: _providers.api_keys_for(name) for name in _providers.CATALOG
}
# เติม alias ให้ resolve_api_key หาเจอด้วย (เช่น "grok" -> XAI_API_KEY)
for _alias, _target in _providers.ALIASES.items():
    if _target in PROVIDER_API_KEYS:
        PROVIDER_API_KEYS.setdefault(_alias, PROVIDER_API_KEYS[_target])

# ตัวแปรที่ระบบรู้จัก (ไว้ใช้ใน check_env / docs)
KNOWN_AI_KEYS = _providers.all_env_keys()


def load_dotenv(path: str | os.PathLike = ".env", override: bool = False,
                verbose: bool = False) -> bool:
    """
    โหลดไฟล์ `.env` เข้า os.environ

    - ถ้ามีไลบรารี python-dotenv จะใช้ตัวนั้น (รองรับ syntax ครบกว่า)
    - ถ้าไม่มี จะใช้ตัว parse แบบง่ายในตัว (รองรับ KEY=VALUE, คอมเมนต์ #, ครอบด้วย " หรือ ')
    - override=False (ดีฟอลต์): ตัวแปรที่มีอยู่แล้วในเชลล์จะไม่ถูกทับ
    คืนค่า: True ถ้าโหลดไฟล์สำเร็จ
    """
    p = Path(path)
    if not p.is_file():
        return False

    try:
        from dotenv import load_dotenv as _ld  # type: ignore
        _ld(dotenv_path=str(p), override=override)
        if verbose:
            print(f"[env] โหลด {p} ด้วย python-dotenv")
        return True
    except ImportError:
        pass  # ไม่มีไลบรารี -> ใช้ตัว parse ในตัว

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # ตัดคอมเมนต์ท้ายบรรทัด (ค่าแบบไม่มี quote) เช่น  KEY=value  # comment
        if " #" in value:
            value = value.split(" #", 1)[0].strip()
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    if verbose:
        print(f"[env] โหลด {p} ด้วยตัว parse ในตัว")
    return True


def apply_env(cfg: Any, env: Optional[Dict[str, str]] = None) -> Any:
    """
    ให้ค่าใน environment variable ทับค่าของ ScanlationConfig (ตาม ENV_MAP)

    - รองรับตัวช่วย "SCANLATION_DEVICE" ตั้งอุปกรณ์เดียวใช้ทั้ง SAM/Inpaint
      (ถ้าไม่ได้ตั้ง SCANLATION_SAM_DEVICE / SCANLATION_INPAINT_DEVICE แยกไว้)
    คืนค่า: cfg (ที่ถูกแก้ไขแล้ว)
    """
    env = env if env is not None else os.environ

    for name, (section, field, cast) in ENV_MAP.items():
        raw = env.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        target = getattr(cfg, section, None)
        if target is None or not hasattr(target, field):
            continue
        try:
            setattr(target, field, cast(raw))
        except (ValueError, TypeError):
            # ค่าแปลงไม่ได้ -> ข้าม (ไม่ให้ทั้งระบบล่มเพราะ env ผิด)
            continue

    # --- ตัวช่วย: อุปกรณ์เดียวใช้ทั้งระบบ ---
    shared_device = env.get("SCANLATION_DEVICE")
    if shared_device:
        if not env.get("SCANLATION_SAM_DEVICE"):
            cfg.detection.sam_device = shared_device
        if not env.get("SCANLATION_INPAINT_DEVICE"):
            cfg.inpainting.device = shared_device

    # --- ตัวช่วย: โปรไฟล์ CPU/GPU ---
    # SCANLATION_PROFILE=cpu  -> ตั้งทุกอย่างให้รันบน CPU ได้โดยไม่ต้องมี GPU
    profile = (env.get("SCANLATION_PROFILE") or "").strip().lower()
    if profile == "cpu":
        cfg.detection.sam_device = "cpu"
        cfg.inpainting.device = "cpu"
        if not env.get("SCANLATION_SAM_MODEL_TYPE"):
            cfg.detection.sam_model_type = "vit_b"
    elif profile in ("gpu", "cuda"):
        cfg.detection.sam_device = "cuda"
        cfg.inpainting.device = "cuda"

    return cfg


def resolve_api_key(backend: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    หา API key ให้ตรงกับ backend ที่เลือก (รองรับหลายเจ้า)
    คืนค่า key ตัวแรกที่เจอ หรือ None
    """
    env = env if env is not None else os.environ
    for name in PROVIDER_API_KEYS.get((backend or "").strip().lower(), []):
        val = env.get(name)
        if val:
            return val
    return None


def env_report(env: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """สรุปว่ามี env ตัวไหนตั้งไว้บ้าง (ใช้โชว์ใน check_env)"""
    env = env if env is not None else os.environ
    app_set = [n for n in ENV_MAP if env.get(n)]
    keys_set = [n for n in KNOWN_AI_KEYS if env.get(n)]
    return {
        "app_vars_set": app_set,
        "ai_keys_set": keys_set,
        "has_dotenv_file": Path(".env").is_file(),
    }
