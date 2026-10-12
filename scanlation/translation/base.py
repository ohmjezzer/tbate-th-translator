# -*- coding: utf-8 -*-
"""
translation/base.py
===================
Interface ของสเตจ Translation + โครงสร้างบริบทการแปล

หลักการ: "Translator ไม่รู้จัก LLM ใด ๆ" — มันรู้แค่ interface
ทำให้สลับผู้ให้บริการ (mock / rule-based / Claude / GPT) ได้โดยไม่กระทบ pipeline
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from ..schemas import Page
from .glossary import StoryContext


@dataclass
class TranslationContext:
    """ข้อมูลประกอบการแปลที่ส่งต่อให้ LLM (บริบทเรื่อง + glossary)"""

    story: StoryContext = field(default_factory=StoryContext)
    page_summary: Optional[str] = None      # สรุปเนื้อหาหน้าก่อนหน้า (ถ้ามี)


class Translator(ABC):
    """แปลง Page (ข้อความต้นฉบับ) → Page (ที่มีคำแปลไทยแล้ว)"""

    @abstractmethod
    def translate_page(self, page: Page, context: Optional[TranslationContext] = None) -> Page:
        raise NotImplementedError

    def translate_pages(self, pages: List[Page],
                        context: Optional[TranslationContext] = None) -> List[Page]:
        """ค่าเริ่มต้น: แปลทีละหน้า (subclass อาจ override เพื่อ batch)"""
        return [self.translate_page(p, context) for p in pages]
