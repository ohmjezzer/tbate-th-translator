# -*- coding: utf-8 -*-
"""
translation/prompts.py
======================
สร้าง System Prompt / User Prompt สำหรับ LLM (Claude / GPT)

หัวใจของการได้ "คำแปลที่เป็นธรรมชาติ" ไม่ใช่แปลตรงตัว:
    1. System prompt ใส่ "บริบทตัวละคร" อย่างชัดเจน (ใครพูด ใช้สรรพนามอะไร)
    2. สอนกติกาภาษาไทย (คำลงท้าย สรรพนาม ระดับภาษา)
    3. สั่งให้ LLM รักษาความหมายเดิม + รักษาคำเฉพาะตาม glossary
    4. บังคับรูปแบบผลลัพธ์เป็น JSON ที่ map ตาม id ของบล็อก (เพื่อเอาไปวางทับได้ตรง)

นี่คือส่วนที่ทำให้ผลลัพธ์ต่างจาก "Google Translate วางทับ" อย่างชัดเจน
"""

from __future__ import annotations

from typing import List

from ..schemas import BlockKind, Bubble

# ---------------------------------------------------------------------------
# System Prompt (ปรับได้ตามเรื่อง/แนว)
# ---------------------------------------------------------------------------
SYSTEM_TEMPLATE = """\
คุณคือ "นักแปลมังงะ/มันฮวาระดับมืออาชีพ" ที่เชี่ยวชาญการแปลอังกฤษ→ไทย
สำหรับงานสแกนเลชัน (scanlation) คุณต้องแปลให้ "อ่านลื่นเหมือนคนไทยเขียนเอง"
ไม่ใช่แปลตรงตัวแบบเครื่องจักร

# บริบทของเรื่อง
{synopsis}

# แนวทางการใช้ภาษา
{style_guide}

# ตัวละครและวิธีพูด (สำคัญมาก — รักษาให้คงเส้นคงวาทั้งเรื่อง)
{characters}

# ศัพท์เฉพาะ / ชื่อเฉพาะ (ต้องใช้คำแปลนี้เท่านั้นทุกครั้งที่พบ)
{glossary}

# กติกาการแปล (ปฏิบัติตามอย่างเคร่งครัด)
1. แปลเป็นภาษาไทย "ธรรมชาติ" เหมาะกับตัวละครที่พูด — ใช้สรรพนาม/คำลงท้ายให้ตรงกับ
   เพศ วัย ฐานะ และความสัมพันธ์ของคู่สนทนา
2. รักษาความหมาย จำนวนข้อมูล และอารมณ์ของต้นฉบับไว้ให้ครบ ห้ามใส่เนื้อหาใหม่เอง
3. ชื่อเฉพาะ/ชื่อคน/สถานที่/คาถา ให้ทับศัพท์ตามที่กำหนดใน glossary
   ถ้าไม่มีใน glossary ให้ทับศัพท์ตามเสียงอ่านอังกฤษ (หรือคงไว้หากเป็นชื่อแบรนด์/URL)
4. ประเภทข้อความ 'sfx' = เสียงเอฟเฟกต์ ให้แปลเป็นคำเลียนเสียงไทยที่กระชับ
   (เช่น "BOOM"→"ตูม", "SLASH"→"ฉับ") ไม่ต้องเป็นประโยค
5. ประเภท 'narration' = ข้อความบรรยาย/กล่องเล่าเรื่อง ให้ใช้ภาษาที่เรียบร้อยเป็นกลาง
6. ประเภท 'speech' = บทสนทนา ให้เป็นภาษาพูดที่ลื่นไหล (จะใส่คำลงท้ายอย่าง
   ครับ/ค่ะ/นะ/สิ/วะ ได้ตามตัวละคร)
7. ถ้าประโยคถูกตัดกลางวลี (มี ... หรือ --) ให้คงการตัดคำนั้นไว้
8. ห้ามใส่คำอธิบายเพิ่ม, ห้ามใส่หมายเหตุ, ห้ามใส่เครื่องหมายคำพูดครอบทั้งประโยค
   นอกจากในต้นฉบับมี
9. ถ้าข้อความต้นฉบับว่างหรือไม่มีความหมาย ให้คงเป็นค่าว่าง

# รูปแบบผลลัพธ์
ตอบกลับเป็น JSON ตาม schema ที่กำหนดเท่านั้น โดยมีฟิลด์:
  "translations": [ {{"id": <string เดิมของบล็อก>, "th": <คำแปลไทย>}}, ... ]
ต้องมีรายการครบทุก id ที่ได้รับ โดยรักษาลำดับเดิมไว้
"""

# ---------------------------------------------------------------------------
# JSON Schema สำหรับ structured output (ใช้ทั้ง Anthropic tool และ OpenAI json_schema)
# ---------------------------------------------------------------------------
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "th": {"type": "string"},
                },
                "required": ["id", "th"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# ตัวสร้าง prompt
# ---------------------------------------------------------------------------
def build_system_prompt(context) -> str:
    """ประกอบ system prompt จาก StoryContext"""
    story = context.story
    return SYSTEM_TEMPLATE.format(
        synopsis=(story.synopsis or "(ไม่มีเรื่องย่อ)").strip(),
        style_guide=(
            story.style_guide
            or "ใช้ภาษาไทยร่วมสมัย อ่านง่าย เป็นธรรมชาติ เน้นบทสนทนาที่มีชีวิตชีวา"
        ).strip(),
        characters=story.character_block(),
        glossary=story.glossary_block(),
    )


def build_user_prompt(bubbles: List[Bubble], target_lang: str = "Thai",
                      page_summary: str | None = None,
                      repair_note: str | None = None) -> str:
    """
    สร้าง user prompt ที่มี "ข้อความทั้งหมดในหน้าเดียว"
    → ให้ LLM เห็นบริบททั้งหน้า จะได้แปลสอดคล้องกัน (เช่น สรรพนามไม่งง)

    repair_note: ใช้ตอน "รอบซ่อม" — แนบคำกำชับพิเศษ (ห้ามเว้น id / ห้ามเหลืออังกฤษ)
    """
    lines: List[str] = []
    if page_summary:
        lines.append(f"[บริบทหน้าก่อน/สรุปเรื่อง] {page_summary}\n")
    if repair_note:
        lines.append(repair_note)
        lines.append("")

    lines.append(f"จงแปลข้อความต่อไปนี้เป็น{target_lang} "
                 f"โดยคงค่า id ไว้ให้ครบทุกบรรทัด:")
    lines.append("")
    for bub in bubbles:
        src = bub.source_text.strip() or bub.translation.strip()
        # ใช้ uid ของ Bubble เป็น id ให้ map กลับได้ตรง
        kind = bub.kind.value if isinstance(bub.kind, BlockKind) else str(bub.kind)
        lines.append(f'id={bub.uid} | ชนิด={kind} | ต้นฉบับ="{src}"')

    lines.append("")
    lines.append("ตอบเป็น JSON ตาม schema เท่านั้น")
    return "\n".join(lines)
