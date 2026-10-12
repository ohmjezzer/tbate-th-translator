# -*- coding: utf-8 -*-
"""
translation/llm_translator.py
=============================
เชื่อมต่อ LLM จริง (Claude 3.5 Sonnet / GPT-4o) พร้อม:
    - LLMBackend (abstract) + AnthropicBackend + OpenAIBackend
    - JSON structured output (ลดความผิดพลาดในการ parse)
    - retry + cache (แปลซ้ำไม่เสียโควตา/เงิน)
    - glossary enforcement หลังแปล

หลักการ DI: LLMTranslator รับ "backend" เข้ามา → ทดสอบด้วย MockBackend ได้
โดยไม่ต้องยิง API จริง
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from ..config import TranslationConfig
from ..schemas import Bubble, Page
from ..utils.logging_utils import get_logger
from .base import Translator, TranslationContext
from .glossary import StoryContext, load_context
from .prompts import OUTPUT_SCHEMA, build_system_prompt, build_user_prompt

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class LLMBackend(ABC):
    """interface ของผู้ให้บริการ LLM — คืนค่าเป็น dict ที่ parse จาก JSON แล้ว"""

    @abstractmethod
    def complete(self, system: str, user: str) -> Dict:
        raise NotImplementedError


class AnthropicBackend(LLMBackend):
    """
    Backend ของ Claude (แนะนำ claude-3-5-sonnet)
    ใช้ "tool use" เพื่อบังคับให้โมเดลคืน JSON ตาม schema อย่างเคร่งครัด
    ต้องตั้ง env: ANTHROPIC_API_KEY
    """

    TOOL_NAME = "submit_translations"

    def __init__(self, config: TranslationConfig, client=None):
        self.cfg = config
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("ต้องติดตั้ง SDK: pip install anthropic") from exc
            self._client = anthropic.Anthropic()   # อ่าน ANTHROPIC_API_KEY อัตโนมัติ

    def complete(self, system: str, user: str) -> Dict:
        tool = {
            "name": self.TOOL_NAME,
            "description": "ส่งผลลัพธ์คำแปลทั้งหมด",
            "input_schema": OUTPUT_SCHEMA,
        }
        resp = self._client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": self.TOOL_NAME},
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == self.TOOL_NAME:
                return block.input
        # fallback: ถ้าโมเดลไม่เรียก tool ให้พยายาม parse text
        text = "".join(getattr(b, "text", "") for b in resp.content)
        return _parse_json(text)


class OpenAIBackend(LLMBackend):
    """
    Backend ของ GPT-4o / GPT-4.1 ใช้ Structured Outputs (json_schema, strict)
    ต้องตั้ง env: OPENAI_API_KEY
    """

    def __init__(self, config: TranslationConfig, client=None):
        self.cfg = config
        if client is not None:
            self._client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("ต้องติดตั้ง SDK: pip install openai") from exc
            self._client = OpenAI()   # อ่าน OPENAI_API_KEY อัตโนมัติ

    def complete(self, system: str, user: str) -> Dict:
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "translations",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                },
            },
        )
        return _parse_json(resp.choices[0].message.content or "")


def _parse_json(text: str) -> Dict:
    """parse JSON แบบทนทาน (ตัด code fence / ข้อความห่อออก)"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


# ---------------------------------------------------------------------------
# Translator หลัก
# ---------------------------------------------------------------------------
class LLMTranslator(Translator):
    """
    แปลทั้งหน้าด้วย LLM 1 ครั้ง (เห็นบริบทครบทั้งหน้า → คำแปลสอดคล้องกัน)

    ตัวอย่าง:
        cfg = TranslationConfig(backend="anthropic")
        tr = LLMTranslator(cfg)                    # โหลด context จากไฟล์ config อัตโนมัติ
        page = tr.translate_page(page)
    """

    def __init__(self, config: Optional[TranslationConfig] = None,
                 backend: Optional[LLMBackend] = None,
                 story: Optional[StoryContext] = None):
        self.cfg = config or TranslationConfig()
        self.backend = backend or self._default_backend()
        self.story = story or load_context(self.cfg)
        self._cache: Dict[str, str] = self._load_cache()

    # ---- สร้าง backend ตาม config ---------------------------------------
    def _default_backend(self) -> LLMBackend:
        if self.cfg.backend == "anthropic":
            return AnthropicBackend(self.cfg)
        if self.cfg.backend == "openai":
            return OpenAIBackend(self.cfg)
        raise ValueError(f"ไม่รู้จัก backend '{self.cfg.backend}' (รองรับ anthropic/openai)")

    # ---- cache -----------------------------------------------------------
    def _cache_path(self) -> Path:
        return Path(self.cfg.cache_path)

    def _load_cache(self) -> Dict[str, str]:
        p = self._cache_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_cache(self) -> None:
        p = self._cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def _key(source: str, kind: str, model: str) -> str:
        raw = f"{model}||{kind}||{source}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    # ---- แปล -------------------------------------------------------------
    def translate_page(self, page: Page,
                       context: Optional[TranslationContext] = None) -> Page:
        ctx = context or TranslationContext(story=self.story)
        bubbles: List[Bubble] = page.bubbles

        # 1) แยกเฉพาะรายการที่ยังไม่มีคำแปล (cache hit ข้ามการเรียก API)
        pending: List[Bubble] = []
        for bub in bubbles:
            src = bub.source_text.strip()
            if not src:
                continue
            cached = self._cache.get(self._key(src, bub.kind.value, self.cfg.model))
            if cached:
                bub.set_translation(cached)
            else:
                pending.append(bub)

        if not pending:
            log.info("หน้า %d: ใช้คำแปลจาก cache ทั้งหมด", page.index)
            return page

        log.info("หน้า %d: เรียก %s (%d บล็อก) ...",
                 page.index, self.cfg.model, len(pending))

        system = build_system_prompt(ctx)
        user = build_user_prompt(pending, target_lang=self.cfg.target_lang,
                                 page_summary=ctx.page_summary)

        result = self._call_with_retry(system, user)
        mapping: Dict[str, str] = {
            str(item["id"]): str(item.get("th", ""))
            for item in result.get("translations", [])
        }

        # 2) เขียนคำแปลกลับ + ตรวจ glossary + เก็บ cache
        for bub in pending:
            th = mapping.get(bub.uid, "").strip()
            if not th:
                log.warning("ไม่พบคำแปลสำหรับ %s — คงข้อความเดิมไว้ก่อน", bub.uid)
                th = bub.source_text
            th = self.story.glossary.enforce(th)
            bub.set_translation(th)
            self._cache[self._key(bub.source_text, bub.kind.value, self.cfg.model)] = th

        self._save_cache()
        return page

    # ---- เรียก backend พร้อม retry --------------------------------------
    def _call_with_retry(self, system: str, user: str) -> Dict:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return self.backend.complete(system, user)
            except Exception as exc:  # noqa: BLE001 - ต้องกันทุก error เพื่อ retry
                last_exc = exc
                wait = 2 ** attempt
                log.warning("เรียก LLM ล้มเหลว (ครั้งที่ %d): %s — รอ %ds",
                            attempt, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"เรียก LLM ล้มเหลวหลัง {self.cfg.max_retries} ครั้ง") from last_exc
