# -*- coding: utf-8 -*-
"""
translation/models.py
=====================
ตัวช่วย "ดูรายชื่อโมเดลที่ใช้ได้จริง" จากผู้ให้บริการ (ไม่ต้องเดา)

เหตุผล: ชื่อโมเดลของผู้ให้บริการเปลี่ยน/เลิกให้บริการบ่อย (เช่น Gemini เปลี่ยนรุ่น)
การ hard-code ชื่อโมเดลจึงมักพัง — ฟังก์ชันนี้ถาม API โดยตรงเพื่อดูว่ามีอะไรใช้ได้

การใช้งาน:
    from scanlation.translation.models import list_models
    names = list_models(cfg)          # -> ["gemini-3.8-flash", ...]

CLI:
    python -m scanlation.cli models
"""

from __future__ import annotations

from typing import List, Optional

from ..config import TranslationConfig
from ..providers import api_keys_for, base_url_for, get as get_provider
from ..utils.logging_utils import get_logger

log = get_logger(__name__)


class ModelListError(RuntimeError):
    """ดึงรายชื่อโมเดลไม่สำเร็จ (เช่น คีย์ผิด/ไม่มีเน็ต)"""


def normalize_model(model: str) -> str:
    """
    ตัดคำนำหน้า 'models/' ออกถ้ามี (บาง endpoint/SDK คืนชื่อแบบมี prefix)
    เช่น 'models/gemini-3.8-flash' -> 'gemini-3.8-flash'
    """
    m = (model or "").strip()
    if m.startswith("models/"):
        m = m[len("models/"):]
    return m


def list_models(cfg: Optional[TranslationConfig] = None, timeout: float = 30.0) -> List[str]:
    """
    คืนรายชื่อโมเดลของ backend ปัจจุบัน (เรียงตามตัวอักษร)

    - backend แบบ OpenAI-compatible (gemini/openai/groq/deepseek/...)
      ใช้ OpenAI SDK: client.models.list()
    - backend anthropic: ใช้ anthropic SDK: client.models.list()
    """
    # รองรับทั้ง TranslationConfig และ ScanlationConfig (มี .translation อยู่ข้างใน)
    if cfg is None:
        cfg = TranslationConfig()
    elif not hasattr(cfg, "backend") and hasattr(cfg, "translation"):
        cfg = cfg.translation  # type: ignore[assignment]
    name = (cfg.backend or "").strip().lower()
    prov = get_provider(name)

    import os

    key = None
    for k in api_keys_for(name):
        if os.environ.get(k):
            key = os.environ[k]
            break

    try:
        if prov is not None and prov.api_style == "anthropic":
            import anthropic

            kwargs = {}
            if cfg.base_url or (prov and prov.base_url):
                kwargs["base_url"] = cfg.base_url or prov.base_url
            if key:
                kwargs["api_key"] = key
            client = anthropic.Anthropic(**kwargs)
            resp = client.models.list()
            ids = [normalize_model(getattr(m, "id", str(m))) for m in resp.data]
            return sorted(set(ids))

        # --- OpenAI-compatible ---
        from openai import OpenAI

        kwargs = {}
        b = cfg.base_url or base_url_for(name)
        if b:
            kwargs["base_url"] = b
        if key:
            kwargs["api_key"] = key
        elif prov is not None and not prov.env_keys:
            kwargs["api_key"] = "ollama"
        client = OpenAI(**kwargs)
        resp = client.models.list()
        ids = [normalize_model(getattr(m, "id", str(m))) for m in resp.data]
        return sorted(set(ids))
    except Exception as exc:  # noqa: BLE001 - แปลงเป็นข้อความช่วยเหลือ
        raise ModelListError(
            f"ดึงรายชื่อโมเดลของ '{name}' ไม่สำเร็จ: {type(exc).__name__}: {exc}"
        ) from exc


def suggest_from_error(err_text: str) -> Optional[str]:
    """
    สกัดชื่อโมเดลที่ผู้ให้บริการ 'แนะนำ' จากข้อความ error
    เช่น "... use models/gemini-3.8-flash for the latest ..." -> "gemini-3.8-flash"
    """
    import re

    if not err_text:
        return None
    # หาคำว่า use <model> หรือ models/<model>
    m = re.search(r"use\s+(?:models/)?([a-zA-Z0-9._\-]+)", err_text)
    if m:
        return m.group(1)
    m = re.search(r"models/([a-zA-Z0-9._\-]+)", err_text)
    if m:
        return m.group(1)
    return None
