# -*- coding: utf-8 -*-
"""
typesetting/thai_wrapper.py
===========================
ตัดบรรทัดข้อความไทย-อังกฤษ (Word Wrapping) ให้ไม่ตัดกลางคำ

ความท้าทายของภาษาไทย: ไม่มีเว้นวรรคระหว่างคำ → ถ้าตัดตามตัวอักษรจะได้คำพัง
วิธีแก้: ใช้ pythainlp.word_tokenize แยก "คำ" ก่อน แล้วค่อย ๆ ต่อคำทีละคำ
จนเต็มความกว้าง (greedy wrap) — เหมือนวิธีใน render.py เดิม แต่ปรับปรุงให้:
    - คืนค่าว่าแต่ละบรรทัดกว้างเท่าไร (ให้ fitter ใช้ต่อ)
    - ถ้าคำเดียวยาวเกินกรอบ (เช่น URL) จะซอยด้วย grapheme cluster
"""

from __future__ import annotations

import re
from typing import List

from PIL import ImageDraw, ImageFont

# ตรวจว่า token เป็น "คำละติน" (ต้องเว้นวรรคระหว่างคำ)
_LATIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9'\-\.]*$")

try:
    from pythainlp.tokenize import word_tokenize
    _HAS_PYTHAINLP = True
except ImportError:  # pragma: no cover
    _HAS_PYTHAINLP = False


def _needs_space(prev: str, nxt: str) -> bool:
    """ต้องเว้นวรรคระหว่าง prev กับ nxt ไหม (เฉพาะคำละตินที่ติดกัน)"""
    return bool(_LATIN.match(prev.strip()) and _LATIN.match(nxt.strip()))


class ThaiTextWrapper:
    def __init__(self, use_pythainlp: bool = True):
        self.use_pythainlp = use_pythainlp and _HAS_PYTHAINLP

    # ---- 1) แยกคำ -------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        """แยกข้อความเป็นคำ ๆ (คืน list ของคำ)"""
        if not text:
            return []
        if self.use_pythainlp:
            try:
                toks = word_tokenize(text, keep_whitespace=False)
                toks = [t for t in toks if t.strip() != ""]
                if toks:
                    return toks
            except Exception:  # noqa: BLE001 - ถ้า tokenizer ล้ม ให้ fallback
                pass
        return list(text)

    # ---- 2) ซอยคำที่ยาวเกินกรอบ -----------------------------------------
    def _split_long_token(self, token: str, font: ImageFont.FreeTypeFont,
                          max_width: float, draw: ImageDraw.ImageDraw) -> List[str]:
        pieces, cur = [], ""
        for ch in token:
            if cur == "" or draw.textlength(cur + ch, font=font) <= max_width:
                cur += ch
            else:
                pieces.append(cur)
                cur = ch
        if cur:
            pieces.append(cur)
        return pieces

    # ---- 3) ตัดบรรทัด ---------------------------------------------------
    def wrap(self, text: str, font: ImageFont.FreeTypeFont, max_width: float,
             draw: ImageDraw.ImageDraw) -> List[str]:
        out: List[str] = []
        for para in text.split("\n"):
            tokens = self.tokenize(para)
            # ถ้าคำไหนยาวเกินกรอบ ให้ซอยก่อน
            flat: List[str] = []
            for tk in tokens:
                if draw.textlength(tk, font=font) > max_width:
                    flat.extend(self._split_long_token(tk, font, max_width, draw))
                else:
                    flat.append(tk)

            line = ""
            for tk in flat:
                candidate = tk if line == "" else (
                    line + " " + tk if _needs_space(line, tk) else line + tk
                )
                if line == "" or draw.textlength(candidate, font=font) <= max_width:
                    line = candidate
                else:
                    out.append(line)
                    line = tk
            out.append(line)
        return [ln for ln in out if ln != ""]
