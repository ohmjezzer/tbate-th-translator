# -*- coding: utf-8 -*-
"""สเตจ Translation: แปลด้วย LLM (Claude 3.5 Sonnet / GPT-4o)"""

from .base import Translator, TranslationContext
from .glossary import CharacterProfile, Glossary, StoryContext, load_context
from .prompts import build_system_prompt, build_user_prompt, OUTPUT_SCHEMA
from .llm_translator import (
    LLMTranslator,
    AnthropicBackend,
    OpenAIBackend,
    LLMBackend,
)

__all__ = [
    "Translator",
    "TranslationContext",
    "CharacterProfile",
    "Glossary",
    "StoryContext",
    "load_context",
    "build_system_prompt",
    "build_user_prompt",
    "OUTPUT_SCHEMA",
    "LLMTranslator",
    "LLMBackend",
    "AnthropicBackend",
    "OpenAIBackend",
]
