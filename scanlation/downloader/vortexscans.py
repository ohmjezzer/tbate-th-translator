# -*- coding: utf-8 -*-
"""
downloader/vortexscans.py
=========================
ChapterSource สำหรับเว็บ vortexscans.org (เว็บตัวอย่างในโปรเจกต์เดิม)

จากการสำรวจจริง (พ.ศ. 2568): หน้า chapter จะฝัง URL รูปทั้งหมดไว้ใน HTML
โดยตรง (ทั้งใน <link rel="preload"> และในข้อมูลที่ฝังมากับหน้า) รูปอยู่บน
โดเมน storage ตามรูปแบบ:

    https://storage.vortexscans.org/upload/series/<slug>/<dir>/page-0001_....webp
    https://storage.vortexscans.org/upload/series/<slug>/<dir>/page-0041_....webp

เราจึงสกัดด้วย regex แล้วเรียงตามเลขหน้า (page-0001 … page-N)
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..config import DownloadConfig
from ..utils.logging_utils import get_logger
from ..utils.registry import register
from .base import ChapterSource, ImageRef

log = get_logger(__name__)

# จับ "รูปหน้า" โดยเฉพาะ (ตัดภาพ featured/cover ออก) — page-<เลข> นำหน้าชื่อไฟล์
_PAGE_URL_RE = re.compile(
    r'https?://[^"\'\s\\]*?/(page-\d+)[^"\'\s\\]*?\.(?:webp|jpg|jpeg|png)',
    re.IGNORECASE,
)


@register("source", "vortexscans")
class VortexScansSource(ChapterSource):
    name = "vortexscans"

    def __init__(self, config: Optional[DownloadConfig] = None):
        self.cfg = config or DownloadConfig()

    # ---- override ค่า default จาก config --------------------------------
    def _user_agent(self) -> str:
        return self.cfg.user_agent

    def _timeout(self) -> float:
        return self.cfg.timeout

    def _retries(self) -> int:
        return self.cfg.retries

    def list_images(self, chapter_url: str) -> List[ImageRef]:
        html = self.fetch_text(chapter_url)
        return self._extract(html, chapter_url)

    # ---- สกัด URL จาก HTML ----------------------------------------------
    def _extract(self, html: str, chapter_url: str = "") -> List[ImageRef]:
        # dedupe คงลำดับ แล้วค่อยเรียงตามเลขหน้า
        seen: dict[str, str] = {}
        for m in _PAGE_URL_RE.finditer(html):
            url = m.group(0)
            label = m.group(1)                 # เช่น "page-0001"
            seen.setdefault(url, label)

        refs: List[ImageRef] = []
        for url, label in seen.items():
            num = self._page_number(label, url)
            refs.append(ImageRef(index=0, url=url, page_label=label,
                                 meta={"page_num": num}))

        # เรียงตามเลขอ้างอิงในชื่อไฟล์ (page-0001 < page-0002 < ...)
        refs.sort(key=lambda r: (r.meta["page_num"], r.url))
        # กำหนด index จริง 1..N
        for i, r in enumerate(refs, start=1):
            r.index = i

        log.info("VortexScans: พบ %d รูปหน้าใน %s", len(refs), chapter_url)
        if not refs:
            log.warning("ไม่พบรูปหน้า — เว็บอาจเปลี่ยนโครงสร้าง หรือ chapter ถูกล็อก")
        return refs

    @staticmethod
    def _page_number(label: str, url: str) -> int:
        m = re.search(r"(\d+)", label)
        if m:
            return int(m.group(1))
        m = re.search(r"page-(\d+)", url)
        return int(m.group(1)) if m else 0
