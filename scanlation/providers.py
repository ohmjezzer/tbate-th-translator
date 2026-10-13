# -*- coding: utf-8 -*-
"""
providers.py
============
แคตตาล็อก "ผู้ให้บริการ LLM" (LLM Providers) — แหล่งข้อมูลกลางของทั้งโปรเจกต์

ทำไมต้องมี:
    - รวม base_url / ชื่อ env var / โมเดลดีฟอลต์ ไว้ที่เดียว (single source of truth)
      แทนที่จะกระจายอยู่หลายไฟล์
    - ระบุชัดว่าผู้ให้บริการใดมี "ระดับฟรี" (free tier) เพื่อให้ผู้ใช้เริ่มต้นได้
      โดยไม่ต้องมีโมเดล/บัตรเครดิตของตัวเอง

การใช้งาน:
    from scanlation.providers import get, list_free, base_url_for, api_keys_for
    p = get("gemini")          # Provider(...)
    p.free                     # True
    base_url_for("groq")       # "https://api.groq.com/openai/v1"

ผู้ให้บริการ "ฟรี" ที่ใช้ได้ทั่วไป (ไม่ต้องใช้บัตรเครดิต — ขอคีย์จากหน้าเว็บได้เลย):
    gemini      Google AI Studio (Gemini) — โควตาฟรีรายวัน, คุณภาพไทยดี
    groq        Groq — เร็วมาก, OpenAI-compatible
    openrouter  OpenRouter — มีโมเดลฟรี (ต่อท้ายด้วย :free)
    cerebras    Cerebras — ฟรี
    mistral     Mistral — มีระดับฟรี
    together    Together AI — มีโมเดลฟรีบางตัว
    github      GitHub Models — ใช้ GitHub PAT
    ollama      Ollama (รันบนเครื่องเอง — ฟรี 100%)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# โครงสร้างข้อมูลของผู้ให้บริการ
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Provider:
    name: str                      # ชื่อสั้น (ใช้ใน SCANLATION_TRANSLATION_BACKEND)
    display: str                   # ชื่อที่แสดงให้ผู้ใช้อ่าน
    api_style: str                 # "anthropic" | "openai"  (รูปแบบ API ที่ใช้เชื่อม)
    base_url: str                  # endpoint (ว่าง = ใช้ค่าดีฟอลต์ของ SDK)
    env_keys: List[str]            # ชื่อ environment variable ที่เก็บ API key (เรียงตามลำดับที่ลอง)
    default_model: str             # โมเดลดีฟอลต์ที่แนะนำ
    free: bool = False             # มีระดับฟรีใช้ได้หรือไม่
    signup_url: str = ""           # หน้าเว็บขอคีย์
    example_models: List[str] = field(default_factory=list)
    notes: str = ""

    def primary_env(self) -> str:
        return self.env_keys[0] if self.env_keys else ""


# ---------------------------------------------------------------------------
# แคตตาล็อก (เรียงจากที่แนะนำให้เริ่มง่ายสุดไปหายาก)
# ---------------------------------------------------------------------------
CATALOG: Dict[str, Provider] = {
    # ---------------------- ผู้ให้บริการ "ฟรี" (แนะนำเริ่มต้น) ----------------------
    "gemini": Provider(
        name="gemini", display="Google Gemini (Google AI Studio)",
        api_style="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        env_keys=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        default_model="gemini-3.8-flash",
        free=True,
        signup_url="https://aistudio.google.com/app/apikey",
        example_models=["gemini-3.8-flash", "gemini-3.8-pro", "gemini-2.5-flash", "gemini-2.5-pro"],
        notes="ขอคีย์ฟรีที่ Google AI Studio (ไม่ต้องใช้บัตรเครดิต) — แนะนำมากสำหรับเริ่มต้น",
    ),
    "groq": Provider(
        name="groq", display="Groq",
        api_style="openai",
        base_url="https://api.groq.com/openai/v1",
        env_keys=["GROQ_API_KEY"],
        default_model="llama-3.3-70b-versatile",
        free=True,
        signup_url="https://console.groq.com/keys",
        example_models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"],
        notes="เร็วมาก + มีโควตาฟรี เหมาะกับงานแปลจำนวนมาก",
    ),
    "openrouter": Provider(
        name="openrouter", display="OpenRouter (รวมโมเดลหลายเจ้า)",
        api_style="openai",
        base_url="https://openrouter.ai/api/v1",
        env_keys=["OPENROUTER_API_KEY"],
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        free=True,
        signup_url="https://openrouter.ai/keys",
        example_models=[
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
            "deepseek/deepseek-r1:free",
        ],
        notes="โมเดลที่ลงท้ายด้วย ':free' ใช้ฟรี — มีหลายรุ่นให้เลือกในคีย์เดียว",
    ),
    "cerebras": Provider(
        name="cerebras", display="Cerebras",
        api_style="openai",
        base_url="https://api.cerebras.ai/v1",
        env_keys=["CEREBRAS_API_KEY"],
        default_model="llama-3.3-70b",
        free=True,
        signup_url="https://cloud.cerebras.ai/",
        example_models=["llama-3.3-70b", "llama3.1-8b"],
        notes="มีระดับฟรี เร็วมาก",
    ),
    "mistral": Provider(
        name="mistral", display="Mistral AI",
        api_style="openai",
        base_url="https://api.mistral.ai/v1",
        env_keys=["MISTRAL_API_KEY"],
        default_model="mistral-small-latest",
        free=True,
        signup_url="https://console.mistral.ai/api-keys/",
        example_models=["mistral-small-latest", "mistral-large-latest", "open-mistral-7b"],
        notes="มีระดับฟรี (La Plateforme)",
    ),
    "together": Provider(
        name="together", display="Together AI",
        api_style="openai",
        base_url="https://api.together.xyz/v1",
        env_keys=["TOGETHER_API_KEY", "TOGETHER_AI_API_KEY"],
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        free=True,
        signup_url="https://api.together.xyz/settings/api-keys",
        example_models=["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"],
        notes="มีโมเดลที่ลงท้าย -Free ให้ใช้ฟรี",
    ),
    "github": Provider(
        name="github", display="GitHub Models",
        api_style="openai",
        base_url="https://models.inference.ai.azure.com",
        env_keys=["GITHUB_TOKEN", "GH_TOKEN"],
        default_model="gpt-4o-mini",
        free=True,
        signup_url="https://github.com/settings/tokens",
        example_models=["gpt-4o-mini", "gpt-4o", "Llama-3.3-70B-Instruct"],
        notes="ใช้ Personal Access Token ของ GitHub — มีโควตาฟรี",
    ),
    "ollama": Provider(
        name="ollama", display="Ollama (รันบนเครื่องเอง)",
        api_style="openai",
        base_url="http://localhost:11434/v1",
        env_keys=[],                      # รันในเครื่อง ไม่ต้องใช้คีย์
        default_model="llama3.1",
        free=True,
        signup_url="https://ollama.com/download",
        example_models=["llama3.1", "qwen2.5", "gemma2"],
        notes="ดาวน์โหลดโมเดลมาไว้ในเครื่อง ฟรี 100% ไม่ต้องมีคีย์ (ต้องรัน ollama serve)",
    ),

    # ---------------------- ผู้ให้บริการระดับเสียเงิน (คุณภาพสูงสุด) ----------------
    "anthropic": Provider(
        name="anthropic", display="Anthropic (Claude)",
        api_style="anthropic",
        base_url="https://api.anthropic.com",
        env_keys=["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
        default_model="claude-3-5-sonnet-latest",
        free=False,
        signup_url="https://console.anthropic.com/settings/keys",
        example_models=["claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"],
        notes="คุณภาพสูงสุด (เสียเงิน)",
    ),
    "openai": Provider(
        name="openai", display="OpenAI (GPT)",
        api_style="openai",
        base_url="https://api.openai.com/v1",
        env_keys=["OPENAI_API_KEY"],
        default_model="gpt-4o",
        free=False,
        signup_url="https://platform.openai.com/api-keys",
        example_models=["gpt-4o", "gpt-4.1", "gpt-4o-mini"],
        notes="คุณภาพสูง (เสียเงิน)",
    ),
    "deepseek": Provider(
        name="deepseek", display="DeepSeek",
        api_style="openai",
        base_url="https://api.deepseek.com/v1",
        env_keys=["DEEPSEEK_API_KEY"],
        default_model="deepseek-chat",
        free=False,
        signup_url="https://platform.deepseek.com/api_keys",
        example_models=["deepseek-chat", "deepseek-reasoner"],
        notes="ราคาถูกมาก",
    ),
    "xai": Provider(
        name="xai", display="xAI (Grok)",
        api_style="openai",
        base_url="https://api.x.ai/v1",
        env_keys=["XAI_API_KEY"],
        default_model="grok-2-latest",
        free=False,
        signup_url="https://console.x.ai/",
        example_models=["grok-2-latest", "grok-beta"],
    ),
    "perplexity": Provider(
        name="perplexity", display="Perplexity",
        api_style="openai",
        base_url="https://api.perplexity.ai",
        env_keys=["PERPLEXITY_API_KEY"],
        default_model="sonar",
        free=False,
        signup_url="https://www.perplexity.ai/settings/api",
        example_models=["sonar", "sonar-pro"],
    ),
    "cohere": Provider(
        name="cohere", display="Cohere",
        api_style="openai",
        base_url="https://api.cohere.ai/compatibility/v1",
        env_keys=["COHERE_API_KEY"],
        default_model="command-r-plus",
        free=False,
        signup_url="https://dashboard.cohere.com/api-keys",
        example_models=["command-r-plus", "command-r"],
    ),
    "fireworks": Provider(
        name="fireworks", display="Fireworks AI",
        api_style="openai",
        base_url="https://api.fireworks.ai/inference/v1",
        env_keys=["FIREWORKS_API_KEY"],
        default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        free=False,
        signup_url="https://fireworks.ai/account/api-keys",
    ),
    "azure": Provider(
        name="azure", display="Azure OpenAI",
        api_style="openai",
        base_url="",   # ต้องตั้งเอง: https://<resource>.openai.azure.com/openai/v1
        env_keys=["AZURE_OPENAI_API_KEY"],
        default_model="gpt-4o",
        free=False,
        signup_url="https://portal.azure.com/",
        notes="ต้องตั้ง SCANLATION_TRANSLATION_BASE_URL เอง + AZURE_OPENAI_ENDPOINT",
    ),
}

# ชื่อย่อที่ยอมรับ (alias) → ชื่อในแคตตาล็อก
ALIASES: Dict[str, str] = {
    "grok": "xai",
    "google": "gemini",
    "az": "azure",
    "llama": "groq",
}


def _canonical(name: str) -> str:
    n = (name or "").strip().lower()
    return ALIASES.get(n, n)


def get(name: str) -> Optional[Provider]:
    """คืน Provider ตามชื่อ (รองรับ alias) หรือ None ถ้าไม่รู้จัก"""
    return CATALOG.get(_canonical(name))


def exists(name: str) -> bool:
    return _canonical(name) in CATALOG


def list_all() -> List[Provider]:
    return list(CATALOG.values())


def list_free() -> List[Provider]:
    """ผู้ให้บริการที่มีระดับฟรี (เรียงตามที่แนะนำให้เริ่ม)"""
    return [p for p in CATALOG.values() if p.free]


def base_url_for(name: str) -> Optional[str]:
    p = get(name)
    return (p.base_url or None) if p else None


def api_keys_for(name: str) -> List[str]:
    p = get(name)
    return list(p.env_keys) if p else []


def default_model_for(name: str) -> Optional[str]:
    p = get(name)
    return p.default_model if p else None


def api_style_for(name: str) -> str:
    p = get(name)
    return p.api_style if p else "openai"


# รวมชื่อ env var ของทุกคีย์ที่ระบบรู้จัก (ไว้ใช้ใน .env.example / check_env)
def all_env_keys() -> List[str]:
    seen: List[str] = []
    for p in CATALOG.values():
        for k in p.env_keys:
            if k not in seen:
                seen.append(k)
    # คีย์เสริมที่ไม่ได้ผูกกับ provider ตรง ๆ แต่ระบบรู้จัก
    for extra in ("HUGGING_FACE_HUB_TOKEN", "HF_TOKEN", "REPLICATE_API_TOKEN",
                  "AZURE_OPENAI_ENDPOINT", "CLOUDFLARE_API_TOKEN",
                  "CLOUDFLARE_ACCOUNT_ID"):
        if extra not in seen:
            seen.append(extra)
    return seen


# ---------------------------------------------------------------------------
# ข้อความช่วยเหลือเมื่อ "ยังไม่มี API key" (ให้ผู้ใช้เริ่มได้ทันที)
# ---------------------------------------------------------------------------
def missing_key_help(backend: str, command: str = "python pipeline_example.py") -> str:
    """
    สร้างข้อความแนะนำแบบเป็นขั้นตอน เมื่อยังไม่มีคีย์ของผู้ให้บริการที่เลือก
    (เรียกใช้จาก pipeline_example.py / cli.py เพื่อ UX ที่ดีขึ้น)
    """
    p = get(backend)
    if p is None:
        return (f"ไม่รู้จัก backend '{backend}'\n"
                f"ดูรายชื่อทั้งหมด:  python -m scanlation.cli setup --list")

    tag = "ฟรี" if p.free else "เสียเงิน"
    out: List[str] = [
        f"ยังไม่มี API key สำหรับ backend '{backend}' "
        f"(ผู้ให้บริการ: {p.display} [{tag}])",
        "",
        "เริ่มใช้งานได้ 3 ทาง เลือกอย่างใดอย่างหนึ่ง (ไม่ต้องมีโมเดลของตัวเอง):",
        "",
        "  1) ทดลองทันทีโดยไม่ต้องใช้คีย์เลย (โหมดเดโม):",
        "       python pipeline_example.py --demo",
        "",
        "  2) ใช้ผู้ให้บริการ AI (มีระดับฟรี — สมัครแล้วได้คีย์เลย):",
    ]
    if p.signup_url:
        out.append(f"       a) ขอคีย์{'ฟรี' if p.free else ''}ได้ที่: {p.signup_url}")
    else:
        out.append("       a) (ผู้ให้บริการนี้ไม่ต้องใช้คีย์ — ตรวจว่าติดตั้ง/รันเซิร์ฟเวอร์แล้ว)")
    out += [
        "       b) สร้างไฟล์ .env แล้วใส่คีย์ด้วยคำสั่งเดียว:",
        f"            python -m scanlation.cli setup --provider {backend}",
    ]
    if p.primary_env():
        out.append(f"          หรือแก้ไฟล์ .env เอง บรรทัด: {p.primary_env()}=<คีย์ของคุณ>")
    out += [
        f"       c) รันใหม่:   {command}",
        "",
        "  3) ดูผู้ให้บริการทั้งหมด (ฟรี/เสียเงิน) หรือเปลี่ยนเจ้า:",
        "       python -m scanlation.cli setup --list",
    ]
    return "\n".join(out)
