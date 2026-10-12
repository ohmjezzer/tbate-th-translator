# -*- coding: utf-8 -*-
"""
downloader/generic.py
=====================
ChapterSource แบบทั่วไป (ใช้กับเว็บอื่นได้โดยไม่ต้องเขียนโค้ด)

1) GenericJsonSource  — อ่านรายการ URL รูปจากไฟล์ JSON (list ของ URL หรือ
   list ของ object ที่มีฟิลด์ url/image/src) เหมาะกับเว็บที่มี API คืน JSON

2) GenericHtmlSource  — ดึงหน้า HTML แล้วจับ URL รูปด้วย regex ที่ปรับได้
   (ผ่าน config.source_regex หรือพารามิเตอร์) เหมาะกับเว็บที่ฝังรูปใน HTML

ทั้งคู่ทำให้ "เพิ่มเว็บใหม่" ทำได้ผ่าน config โดยไม่ต้องแตะโค้ด pipeline
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from ..config import DownloadConfig
from ..utils.logging_utils import get_logger
from ..utils.registry import register
from .base import ChapterSource, ImageRef

log = get_logger(__name__)

# regex เริ่มต้นสำหรับ GenericHtmlSource: จับ URL รูปภาพทั่วไป
_DEFAULT_IMG_RE = re.compile(
    r'https?://[^"\'\s\\<>]+?\.(?:webp|jpg|jpeg|png)(?:\?[^"\'\s\\<>]*)?',
    re.IGNORECASE,
)

# คีย์ที่อาจใช้เก็บ URL รูปใน JSON ต่างๆ
_URL_KEYS = ("url", "image", "image_url", "src", "link", "path")


@register("source", "generic-json")
class GenericJsonSource(ChapterSource):
    """อ่านรายการ URL จากไฟล์ JSON (หรือ URL ที่คืน JSON)"""

    name = "generic-json"

    def __init__(self, config: Optional[DownloadConfig] = None,
                 list_file: Optional[str] = None, url_key: Optional[str] = None):
        self.cfg = config or DownloadConfig()
        self.list_file = list_file or self.cfg.image_urls_file
        self.url_key = url_key

    def list_images(self, chapter_url: str = "") -> List[ImageRef]:
        if not self.list_file:
            raise ValueError("GenericJsonSource ต้องระบุ image_urls_file (ไฟล์ JSON)")
        raw = Path(self.list_file).read_text(encoding="utf-8")
        data = json.loads(raw)
        urls = self._collect_urls(data)
        refs = [ImageRef(index=i, url=u, page_label=f"page-{i:04d}")
                for i, u in enumerate(urls, start=1)]
        log.info("GenericJson: พบ %d รูป", len(refs))
        return refs

    def _collect_urls(self, data) -> List[str]:
        """รองรับทั้ง list[str] และ list[dict] (หาคีย์ url/image/src)"""
        out: List[str] = []
        def walk(node):
            if isinstance(node, str):
                if re.search(r"\.(webp|jpg|jpeg|png)(\?|$)", node, re.I):
                    out.append(node)
            elif isinstance(node, list):
                for x in node:
                    walk(x)
            elif isinstance(node, dict):
                # ใช้คีย์ที่กำหนดก่อน ถ้าไม่เจอค่อยไล่คีย์มาตรฐาน
                keys = ([self.url_key] if self.url_key else []) + list(_URL_KEYS)
                used = False
                for k in keys:
                    if k and k in node and isinstance(node[k], str):
                        walk(node[k]); used = True; break
                if not used:
                    for v in node.values():
                        walk(v)
        walk(data)
        # dedupe คงลำดับ
        return list(dict.fromkeys(out))


@register("source", "generic-html")
class GenericHtmlSource(ChapterSource):
    """ดึงหน้า HTML แล้วจับ URL รูปด้วย regex (ปรับได้ผ่าน source_regex)"""

    name = "generic-html"

    def __init__(self, config: Optional[DownloadConfig] = None,
                 source_regex: Optional[str] = None):
        self.cfg = config or DownloadConfig()
        self.source_regex = source_regex

    def _user_agent(self) -> str:
        return self.cfg.user_agent

    def _timeout(self) -> float:
        return self.cfg.timeout

    def _retries(self) -> int:
        return self.cfg.retries

    def list_images(self, chapter_url: str) -> List[ImageRef]:
        html = self.fetch_text(chapter_url)
        pattern = re.compile(self.source_regex, re.IGNORECASE) if self.source_regex else _DEFAULT_IMG_RE
        urls = list(dict.fromkeys(m.group(0) for m in pattern.finditer(html)))
        refs = [ImageRef(index=i, url=u, page_label=f"page-{i:04d}")
                for i, u in enumerate(urls, start=1)]
        log.info("GenericHtml: พบ %d รูป", len(refs))
        return refs
