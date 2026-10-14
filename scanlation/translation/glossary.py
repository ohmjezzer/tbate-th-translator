# -*- coding: utf-8 -*-
"""
translation/glossary.py
=======================
ระบบ "บริบทเรื่อง" (Story Context) ที่ทำให้คำแปลมีมิติของตัวละคร

ปัญหาของการแปลแบบเดิม (แปลทีละประโยค): LLM ไม่รู้ว่า "ใครพูดกับใคร"
ทำให้สรรพนาม (ฉัน/ข้า/เจ้า/คุณ/เธอ) และน้ำเสียงไม่คงเส้นคงวา

โซลูชัน: ป้อน 3 อย่างเข้า system prompt
    1. CharacterProfile — ตัวละครแต่ละคน พูดยังไง ใช้สรรพนามอะไร
    2. Glossary         — คำศัพท์เฉพาะ/ชื่อเฉพาะ ที่ต้องแปลเหมือนกันทั้งเรื่อง
    3. synopsis         — เรื่องย่อ เพื่อให้ LLM เข้าใจบริบท

ไฟล์ config ตัวอย่าง: config/characters.json, config/glossary.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CharacterProfile:
    """โปรไฟล์การพูดของตัวละคร 1 ตัว"""

    name: str                                   # ชื่อ (ใช้เรียกใน prompt)
    aliases: List[str] = field(default_factory=list)   # ชื่อเรียกอื่น/ฉายา
    description: str = ""                       # บทบาท/นิสัยสั้น ๆ
    pronouns: str = ""                          # สรรพนามแทนตัวเอง เช่น "ฉัน", "ข้า", "เจ้า(แทนตัวเองแบบหยาบ)"
    address_others: str = ""                    # วิธีเรียกคนอื่น เช่น "เธอ", "เจ้า", "ท่าน"
    speech_style: str = ""                      # น้ำเสียง เช่น "สุภาพ ทางการ", "กวนประสาท", "ดุดัน"

    def to_prompt_line(self) -> str:
        parts = [f"- {self.name}"]
        if self.aliases:
            parts.append(f"(เรียกอีกชื่อ: {', '.join(self.aliases)})")
        meta = []
        if self.description:
            meta.append(self.description)
        if self.pronouns:
            meta.append(f"แทนตัวเองว่า \"{self.pronouns}\"")
        if self.address_others:
            meta.append(f"เรียกคู่สนทนาว่า \"{self.address_others}\"")
        if self.speech_style:
            meta.append(f"สไตล์การพูด: {self.speech_style}")
        return parts[0] + " — " + " ; ".join(meta)


class Glossary:
    """พจนานุกรมคำเฉพาะ: source term -> คำแปลไทยที่ "ตรึง" ไว้"""

    def __init__(self, entries: Optional[Dict[str, str]] = None):
        self.entries: Dict[str, str] = dict(entries or {})

    def add(self, source: str, target: str) -> None:
        self.entries[source] = target

    def to_prompt_lines(self) -> List[str]:
        return [f"- \"{k}\" → \"{v}\"" for k, v in self.entries.items()]

    def enforce(self, text: str) -> str:
        """
        Post-check: ถ้า LLM แปลคำเฉพาะผิด ลองบังคับแทนด้วยคำที่กำหนดไว้
        (ใช้เท่าที่ปลอดภัย — เตือนผู้ใช้ให้ตรวจซ้ำอีกครั้ง)

        ★ แทนคำที่ "ยาวก่อน" เสมอ (longest-first)

        ทำไมสำคัญ: เดิมไล่แทนตามลำดับใน dict ทำให้คำที่ซ้อนกันอย่าง
        "Mana" กับ "Mana Core" ถูกแทนผิดลำดับ -> เหลือ "มานา Core" ซึ่งเป็น
        ไทยปนอังกฤษ (บั๊กที่พบใน sandbox) เรียงยาว->สั้นทำให้ "Mana Core"
        ถูกแทนทั้งวลีก่อน แล้ว "Mana" ที่เหลือจึงค่อยถูกแทนทีหลัง
        """
        for src in sorted(self.entries, key=len, reverse=True):
            th = self.entries[src]
            if src and src in text and th not in text:
                text = text.replace(src, th)
        return text

    def allowed_latin(self) -> List[str]:
        """
        คำอังกฤษที่ "อนุญาต" ให้เหลือในคำแปลได้ = คำศัพท์เฉพาะที่มีค่าแปลเป็นอังกฤษเอง
        (เช่นชื่อเว็บ "Vortex Scans") — ใช้ตรวจว่า "ไทยปนอังกฤษ" จริงหรือไม่
        """
        out = []
        for src, th in self.entries.items():
            if th and all(not ("ก" <= c <= "๙") for c in th):
                out.append(th)
        return out

    @staticmethod
    def from_json(path: str | Path) -> "Glossary":
        p = Path(path)
        if not p.exists():
            return Glossary()
        return Glossary(json.loads(p.read_text(encoding="utf-8")))


@dataclass
class StoryContext:
    """รวมบริบททั้งหมดของเรื่อง"""

    synopsis: str = ""
    characters: List[CharacterProfile] = field(default_factory=list)
    glossary: Glossary = field(default_factory=Glossary)
    style_guide: str = ""        # แนวทางรวม เช่น "ใช้ภาษาพูดธรรมชาติ ไม่ทางการ"

    # ---- helper ----------------------------------------------------------
    def character_block(self) -> str:
        if not self.characters:
            return "(ไม่มีข้อมูลตัวละคร)"
        return "\n".join(c.to_prompt_line() for c in self.characters)

    def glossary_block(self) -> str:
        lines = self.glossary.to_prompt_lines()
        return "\n".join(lines) if lines else "(ไม่มีศัพท์เฉพาะ)"

    @staticmethod
    def from_json(characters_path: Optional[str] = None,
                  glossary_path: Optional[str] = None,
                  synopsis: str = "",
                  style_guide: str = "") -> "StoryContext":
        characters: List[CharacterProfile] = []
        if characters_path and Path(characters_path).exists():
            data = json.loads(Path(characters_path).read_text(encoding="utf-8"))
            if isinstance(data, dict) and "characters" in data:
                synopsis = data.get("synopsis", synopsis)
                style_guide = data.get("style_guide", style_guide)
                data = data["characters"]
            for item in data:
                if isinstance(item, str):           # รองรับชื่อล้วน ๆ
                    characters.append(CharacterProfile(name=item))
                else:
                    characters.append(CharacterProfile(**item))
        glossary = Glossary.from_json(glossary_path) if glossary_path else Glossary()
        return StoryContext(synopsis=synopsis, characters=characters,
                            glossary=glossary, style_guide=style_guide)

    def to_dict(self) -> Dict:
        return {
            "synopsis": self.synopsis,
            "style_guide": self.style_guide,
            "characters": [asdict(c) for c in self.characters],
            "glossary": self.glossary.entries,
        }


def load_context(config) -> StoryContext:
    """สร้าง StoryContext จาก TranslationConfig โดยตรง"""
    return StoryContext.from_json(
        characters_path=config.characters_path,
        glossary_path=config.glossary_path,
    )
