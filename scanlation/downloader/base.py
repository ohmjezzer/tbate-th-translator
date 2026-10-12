# -*- coding: utf-8 -*-
"""
downloader/base.py
==================
Interface ของสเตจ Download & Ingest

แนวคิด: แยก "การหา URL รูป" ออกจาก "การดาวน์โหลด/แปลง/บันทึก"
    - ChapterSource   : บอกได้ว่า chapter หนึ่งมีรูปอะไรบ้าง (ดึงจาก HTML/API/JSON)
    - ImageConverter  : บังคับแปลงภาพทุกฟอร์แมต → WebP (ตามความต้องการ)
    - ChapterDownloader : ตัวประสาน (orchestrate) ทั้งหมด → ไฟล์ใน input_dir

ทำให้เพิ่มเว็บใหม่ได้เพียงเขียน ChapterSource ตัวใหม่ (Open/Closed Principle)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ImageRef:
    """อ้างอิงรูปภาพ 1 หน้า (ก่อนดาวน์โหลด)"""

    index: int                 # ลำดับหน้า (1-based) ใช้กำหนดชื่อไฟล์
    url: str                   # URL ต้นทาง
    page_label: str = ""       # ป้ายชื่อหน้าเดิมจากเว็บ (เพื่อการดีบัก)
    meta: Dict = field(default_factory=dict)


class ChapterSource(ABC):
    """
    ตัวดึง "รายการรูป" ของ chapter หนึ่งจากเว็บ

    implement:
        list_images(chapter_url) -> List[ImageRef]   (เรียงตาม index แล้ว)
    """

    name: str = "base"

    @abstractmethod
    def list_images(self, chapter_url: str) -> List[ImageRef]:
        raise NotImplementedError

    # helper ที่มักใช้ซ้ำ: HTTP GET แบบมี retry/user-agent (ให้ subclass ใช้ร่วม)
    def fetch_text(self, url: str) -> str:
        import requests

        from ..utils.logging_utils import get_logger

        log = get_logger(__name__)
        headers = {"User-Agent": self._user_agent()}
        last_exc = None
        for attempt in range(1, self._retries() + 1):
            try:
                r = requests.get(url, headers=headers, timeout=self._timeout())
                r.raise_for_status()
                return r.text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("ดึง %s ล้มเหลว (ครั้งที่ %d): %s", url, attempt, exc)
        raise RuntimeError(f"ดึงหน้าเว็บไม่สำเร็จ: {url}") from last_exc

    # ค่า default เหล่านี้จะถูก override โดย subclass ที่ผูกกับ DownloadConfig
    def _user_agent(self) -> str:
        return "Mozilla/5.0"

    def _timeout(self) -> float:
        return 60.0

    def _retries(self) -> int:
        return 3
