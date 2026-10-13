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

การแก้ปัญหา "บางบอลลูนไม่ถูกแปล / ไทยปนอังกฤษ" (ดู _verify + _repair):
    - แบ่งเป็นชุดย่อย (batch) ตามจำนวนบล็อก กัน LLM ตอบยาวจน JSON ขาดกลางคัน
    - ตรวจผลลัพธ์ว่าครบทุก id และไม่เหลือคำอังกฤษค้าง
    - ส่ง "รอบซ่อม" (repair pass) เฉพาะรายการที่มีปัญหา พร้อมกำชับเป็นพิเศษ
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import TranslationConfig
from ..providers import api_style_for, base_url_for, exists as provider_exists, list_free
from ..schemas import BlockKind, Bubble, Page
from ..utils.logging_utils import get_logger
from .base import Translator, TranslationContext
from .glossary import StoryContext, load_context
from .models import normalize_model, suggest_from_error
from .prompts import OUTPUT_SCHEMA, build_system_prompt, build_user_prompt

log = get_logger(__name__)


def _provider_base_url(backend: str) -> Optional[str]:
    """คืน base_url ที่เหมาะกับ backend (จากแคตตาล็อกกลาง) หรือ None"""
    return base_url_for(backend)


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
            # อ่าน ANTHROPIC_API_KEY อัตโนมัติ; รองรับ base_url ทางเลือก (proxy/self-host)
            kwargs = {}
            if self.cfg.base_url:
                kwargs["base_url"] = self.cfg.base_url
            key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            if key:
                kwargs["api_key"] = key
            self._client = anthropic.Anthropic(**kwargs)

    def complete(self, system: str, user: str) -> Dict:
        tool = {
            "name": self.TOOL_NAME,
            "description": "ส่งผลลัพธ์คำแปลทั้งหมด",
            "input_schema": OUTPUT_SCHEMA,
        }
        resp = self._client.messages.create(
            model=normalize_model(self.cfg.model),
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

    รองรับผู้ให้บริการที่ใช้ API แบบ OpenAI-compatible ด้วย (ผ่าน base_url):
    DeepSeek / Groq / OpenRouter / Together / Mistral / xAI(Grok) / Ollama ฯลฯ
    โดยกำหนด ``translation.base_url`` + ``translation.model`` ให้ตรง
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
            self._client = self._make_client(OpenAI)

    def _make_client(self, OpenAI):
        """สร้าง client โดยดึง api_key/base_url ให้ตรงกับ backend ที่เลือก"""
        from ..env import resolve_api_key
        from ..providers import api_keys_for, get as get_provider

        prov = get_provider(self.cfg.backend)
        kwargs: Dict = {}
        base_url = self.cfg.base_url or _provider_base_url(self.cfg.backend)
        if base_url:
            kwargs["base_url"] = base_url

        key = resolve_api_key(self.cfg.backend)
        if key:
            kwargs["api_key"] = key
        elif prov is not None and not prov.env_keys:
            # ผู้ให้บริการที่ไม่ต้องใช้คีย์ (เช่น ollama ที่รันในเครื่อง)
            kwargs["api_key"] = "ollama"
        elif prov is not None and prov.env_keys:
            raise RuntimeError(
                f"ไม่พบ API key สำหรับ '{self.cfg.backend}' "
                f"({', '.join(api_keys_for(self.cfg.backend))}) — "
                f"ขอคีย์{'ฟรี' if prov.free else ''}ได้ที่ {prov.signup_url} "
                f"แล้วใส่ใน .env (หรือรัน: python -m scanlation.cli setup --provider {self.cfg.backend})"
            )
        return OpenAI(**kwargs)

    def complete(self, system: str, user: str) -> Dict:
        resp = self._client.chat.completions.create(
            model=normalize_model(self.cfg.model),
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
        name = (self.cfg.backend or "").strip().lower()
        style = api_style_for(name)
        if style == "anthropic":
            return AnthropicBackend(self.cfg)
        if provider_exists(name):
            # ผู้ให้บริการที่ใช้ API แบบ OpenAI-compatible (openai/gemini/groq/deepseek/...)
            return OpenAIBackend(self.cfg)
        free_names = ", ".join(p.name for p in list_free())
        raise ValueError(
            f"ไม่รู้จัก backend '{self.cfg.backend}' — "
            f"ดูรายชื่อทั้งหมด: python -m scanlation.cli setup --list   "
            f"(ผู้ให้บริการฟรีที่ใช้ได้: {free_names})"
        )

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
                # ไม่มีข้อความต้นฉบับ -> ไม่มีอะไรให้แปล
                # แต่ต้อง "ทำเครื่องหมาย" ไว้ เพื่อให้ stages ถัดไปรู้ว่าไม่มีคำแปล
                bub.meta["untranslated"] = True
                continue
            if not self._is_translatable(src, bub):
                # ★ ต้นฉบับเป็น "ขยะจาก OCR" (เช่น "... . . . | |") -> ไม่มีความหมายให้แปล
                #   เดิมส่งเข้า LLM ทำให้ LLM ตอบค่าว่าง แล้วโค้ด fallback เอา "ขยะ" มาแสดง
                log.warning("%s: ต้นฉบับไม่มีความหมายให้แปล %r — ข้าม (mark untranslated)",
                            bub.uid, src[:40])
                bub.meta["untranslated"] = True
                bub.meta["source_text"] = bub.source_text
                continue
            cached = self._cache.get(self._key(src, bub.kind.value, self.cfg.model))
            if cached:
                bub.set_translation(cached)
            else:
                pending.append(bub)

        if not pending:
            log.info("หน้า %d: ใช้คำแปลจาก cache ทั้งหมด", page.index)
            return page

        # ★ แบ่งเป็นชุดย่อย เพื่อกัน LLM ตอบยาวเกิน max_tokens จน JSON ขาดกลางคัน
        #   (ต้นเหตุอันดับหนึ่งของ "บางบอลลูนไม่ถูกแปล")
        batch_size = max(1, int(getattr(self.cfg, "batch_size", 12)))
        batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
        log.info("หน้า %d: เรียก %s (%d บล็อก, %d ชุด) ...",
                 page.index, self.cfg.model, len(pending), len(batches))

        system = build_system_prompt(ctx)
        for bi, batch in enumerate(batches, 1):
            mapping = self._translate_batch(system, batch, ctx)
            # ★ ตรวจผลลัพธ์ + ซ่อม (รอบซ่อมยิงเฉพาะรายการที่มีปัญหา)
            mapping = self._verify_and_repair(system, batch, mapping, ctx)

            for bub in batch:
                self._apply(bub, mapping.get(bub.uid, ""))

        self._save_cache()
        return page

    # ---- แปล 1 ชุด + parse ------------------------------------------------
    def _translate_batch(self, system: str, batch: List[Bubble],
                         ctx: TranslationContext) -> Dict[str, str]:
        user = build_user_prompt(batch, target_lang=self.cfg.target_lang,
                                 page_summary=ctx.page_summary)
        result = self._call_with_retry(system, user)
        return {
            str(item.get("id", "")).strip(): str(item.get("th", ""))
            for item in (result.get("translations") or [])
        }

    # ---- ตรวจ + ซ่อมรอบที่ 1: id ที่ขาด, คำแปลว่าง, ไทยปนอังกฤษ -------------
    def _verify_and_repair(self, system: str, batch: List[Bubble],
                           mapping: Dict[str, str],
                           ctx: TranslationContext) -> Dict[str, str]:
        """
        ตรวจว่า "ทุกบอลลูนได้คำแปลที่ใช้ได้" แล้วหรือยัง

        ปัญหาที่ตรวจ (เรียงตามที่เจอจริงใน sandbox):
            1. ไม่มี id นี้ในคำตอบ       -> เดิม "คงข้อความอังกฤษเดิม" (ต้นตอบอลลูนเป็นอังกฤษ)
            2. คำแปลเป็นค่าว่าง ""        -> เดิมก็ fallback เป็นอังกฤษเหมือนกัน
            3. คำแปลยังมีคำอังกฤษค้าง    -> ได้ "ไทยปนอังกฤษ"
            4. id ผิดรูปแบบ (#bub0, 0)   -> ทำให้ทั้ง 1-3 เกิดพร้อมกันทั้งหน้า
        """
        allow_latin = [x for x in self.story.glossary.allowed_latin()]
        needs: List[Bubble] = []
        for bub in batch:
            th = (mapping.get(bub.uid) or "").strip()
            if not th:
                needs.append(bub)
            elif self._has_leftover_english(th, allow_latin):
                needs.append(bub)

        passes = max(0, int(getattr(self.cfg, "repair_passes", 1)))
        for attempt in range(1, passes + 1):
            if not needs:
                break
            log.warning("หน้า: มี %d บล็อกที่คำแปลไม่สมบูรณ์ — ซ่อมรอบที่ %d (%s)",
                        len(needs), attempt,
                        ", ".join(b.uid for b in needs[:8]))
            # ส่งเฉพาะรายการที่มีปัญหา + กำชับเรื่องที่มักพลาด
            user = build_user_prompt(
                needs, target_lang=self.cfg.target_lang,
                page_summary=(ctx.page_summary or ""),
                repair_note=(
                    "REPAIR: ด้านล่างนี้คือรายการที่ยังมีปัญหา คำตอบก่อนหน้าขาด id นี้ "
                    "หรือยังเหลือคำภาษาอังกฤษค้างอยู่ในคำแปล\n"
                    "จงตอบ “ครบทุก id” ด้านล่าง และเขียนคำแปลเป็น “ภาษาไทยล้วน” "
                    "(ห้ามเหลือคำอังกฤษ ยกเว้นชื่อเฉพาะที่กำหนดใน glossary) "
                    "ถ้าไม่แน่ใจความหมายของวลีอังกฤษ ให้แปลตามบริบทให้กลายเป็นไทย "
                    "และต้องไม่ตอบเป็นค่าว่าง"
                ),
            )
            try:
                result = self._call_with_retry(system, user)
            except Exception as exc:  # noqa: BLE001
                log.warning("ซ่อมคำแปลล้มเหลว: %s", exc)
                break
            got = {str(i.get("id", "")).strip(): str(i.get("th", ""))
                   for i in (result.get("translations") or [])}
            still: List[Bubble] = []
            for bub in needs:
                th = (got.get(bub.uid) or "").strip()
                if th and not self._has_leftover_english(th, allow_latin):
                    mapping[bub.uid] = th
                else:
                    if th:
                        mapping[bub.uid] = th      # ดีกว่าเดิม แม้ยังปนอังกฤษ
                    still.append(bub)
            needs = still
        return mapping

    @staticmethod
    def _is_translatable(src: str, bub: Bubble) -> bool:
        """
        มี "เนื้อความให้แปล" จริงไหม

        - SFX: ยอมรับแม้ไม่ใช่ตัวอักษร (เช่น "!?" "—") เพราะเป็นเสียง/สัญลักษณ์
        - ชนิดอื่น: ต้องมีตัวอักษรอย่างน้อย 1 ตัว (ไทย/อังกฤษ/อื่น ๆ)
          ถ้าเป็นแต่เครื่องหมายวรรคตอน/จุดไข่ปลา = ขยะจาก OCR -> ไม่ต้องแปล
        """
        if bub.kind == BlockKind.SFX:
            return True
        return any(ch.isalpha() for ch in src)

    @staticmethod
    def _has_leftover_english(text: str, allow_latin: Sequence[str] = ()) -> bool:
        """มี 'คำอังกฤษ' ค้างในคำแปลไทยไหม (ยอมรับคำที่อนุญาต + อักษรโรมันในชื่อเฉพาะ)"""
        for allowed in allow_latin:
            text = text.replace(allowed, " ")
        # คำอังกฤษ = มีตัวอักษรละตินติดกัน >= 2 ตัวขึ้นไป (กันกรณี "I" เดี่ยว ๆ)
        return re.search(r"[A-Za-z]{2,}", text) is not None

    def _apply(self, bub: Bubble, th: str) -> None:
        """
        เขียนคำแปลกลับเข้าบอลลูน

        ★ เปลี่ยนพฤติกรรมจากเดิม:
          เดิมเมื่อ LLM ตอบไม่ครบ id โค้ดจะตั้ง th = source_text **แล้วยังเอา
          enforce() ของ glossary ไปรันกับข้อความอังกฤษนั้น** → ได้ข้อความ
          "ไทยปนอังกฤษ" (เช่น 'The มานา Core is unstable, Arthur.')
          ซึ่งเป็นอาการที่ผู้ใช้เห็น ตอนนี้แยกสองเส้นทางชัดเจน:

            - ได้คำแปลจริง  -> enforce() ได้ตามปกติ
            - ไม่ได้คำแปล    -> ทำตาม cfg.on_missing
                 "keep_source" (ค่าดีฟอลต์) = คงข้อความต้นฉบับ "อังกฤษล้วน"
                                 (ไม่ผ่าน glossary จึงไม่กลายเป็นไทยปนอังกฤษ)
                 "blank"                     = เว้นว่างไว้ (ไม่มีอะไรถูกวาดทับ)
              ทั้งสองกรณีทำเครื่องหมาย meta["untranslated"]=true + meta["source_text"]
              เพื่อให้หน้า preview / ขั้นต่อไปรู้ว่าบอลลูนนี้ยังต้องแก้
        """
        th = (th or "").strip()
        if not th:
            bub.meta["untranslated"] = True
            bub.meta["source_text"] = bub.source_text
            on_missing = (getattr(self.cfg, "on_missing", "keep_source") or "keep_source").lower()
            if on_missing == "blank":
                bub.set_translation("")
                log.warning("ไม่พบคำแปลสำหรับ %s — เว้นว่าง (meta.untranslated=true)", bub.uid)
            else:
                # คงข้อความเดิมแบบ "อังกฤษล้วน" (ไม่ผ่าน glossary) เพื่อไม่ให้เป็นไทยปนอังกฤษ
                bub.set_translation(bub.source_text)
                log.warning("ไม่พบคำแปลสำหรับ %s — คงข้อความต้นฉบับไว้ "
                            "(meta.untranslated=true; แก้ได้ในหน้า preview หรือเปิด "
                            "repair_passes เพิ่ม)", bub.uid)
            return

        # glossary บังคับคำเฉพาะ — "เฉพาะกรณีได้คำแปลจริง"
        th = self.story.glossary.enforce(th)
        if self._has_leftover_english(th, self.story.glossary.allowed_latin()):
            log.warning("%s: คำแปลยังมีอังกฤษปนอยู่: %r", bub.uid, th)
            bub.meta["leftover_english"] = True
        bub.meta.pop("untranslated", None)
        bub.set_translation(th)
        self._cache[self._key(bub.source_text, bub.kind.value, self.cfg.model)] = th

    # ---- ตรวจว่า error เกิดจาก "โมเดลใช้ไม่ได้" (ไม่ควร retry) -----------
    def _model_error_hint(self, exc: Exception) -> Optional[str]:
        """
        ถ้า error บอกว่าโมเดลเลิกให้บริการ/ไม่พบ -> คืนข้อความช่วยเหลือ
        (พร้อมชื่อโมเดลที่ผู้ให้บริการแนะนำ + วิธีดูรายชื่อโมเดลจริง)
        """
        text = f"{exc}"
        low = text.lower()
        is_missing = any(s in low for s in (
            "no longer available", "not_found", "not found", "does not exist",
            "unsupported model", "unknown model", "invalid model", "deprecated",
        ))
        if not is_missing:
            return None
        suggested = suggest_from_error(text)
        lines = [
            f"โมเดล '{self.cfg.model}' ใช้ไม่ได้ (ผู้ให้บริการแจ้ง: ไม่พบ/เลิกให้บริการ)",
        ]
        if suggested:
            lines.append(f"  ผู้ให้บริการแนะนำให้ใช้: {suggested}")
        lines += [
            f"  รายชื่อโมเดลที่ใช้ได้จริง:  python -m scanlation.cli models",
            f"  แก้ไขได้โดยตั้งค่าใน .env:  SCANLATION_TRANSLATION_MODEL={suggested or '<ชื่อโมเดล>'}"
            f"   (backend ปัจจุบัน: {self.cfg.backend})",
        ]
        return "\n".join(lines)

    # ---- เรียก backend พร้อม retry --------------------------------------
    def _call_with_retry(self, system: str, user: str) -> Dict:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return self.backend.complete(system, user)
            except Exception as exc:  # noqa: BLE001 - ต้องกันทุก error เพื่อ retry
                last_exc = exc
                # ถ้าเป็นปัญหา "ชื่อโมเดล" ให้แจ้งทันที ไม่ต้อง retry ให้เสียเวลา
                hint = self._model_error_hint(exc)
                if hint:
                    raise RuntimeError(hint) from exc
                wait = 2 ** attempt
                log.warning("เรียก LLM ล้มเหลว (ครั้งที่ %d): %s — รอ %ds",
                            attempt, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"เรียก LLM ล้มเหลวหลัง {self.cfg.max_retries} ครั้ง") from last_exc
