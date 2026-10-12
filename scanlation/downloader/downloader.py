# -*- coding: utf-8 -*-
"""
downloader/downloader.py
========================
ChapterDownloader — ตัวประสานสเตจ Download & Ingest

    chapter_url ──(ChapterSource)──► ImageRef[] ──(HTTP)──► bytes
                                                        │
                                          (ImageConverter) ▼ บังคับ WebP
                                                    orig/page-0001.webp …
                                                        │
                                             download_manifest.json

ผลลัพธ์: ไฟล์ WebP ที่ตั้งชื่อตาม pattern `page-0001.webp` เรียงตามลำดับหน้า
วางใน cfg.out_dir (ดีฟอลต์ = paths.input_dir เช่น orig/) พร้อมให้
ScanlationPipeline.discover_pages() อ่านต่อได้ทันที
"""

from __future__ import annotations

import hashlib
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

from ..config import DownloadConfig
from ..utils.logging_utils import get_logger
from ..utils.registry import build, available
from .base import ChapterSource, ImageRef
from .converter import ImageConverter

log = get_logger(__name__)


@dataclass
class DownloadReport:
    chapter_url: str
    source: str
    out_dir: str
    total: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    files: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"[download] {self.source}: {self.downloaded}/{self.total} หน้า "
                f"(ข้าม {self.skipped}, ล้มเหลว {self.failed}) → {self.out_dir}")


class ChapterDownloader:
    def __init__(self, config: Optional[DownloadConfig] = None,
                 source: Optional[ChapterSource] = None):
        self.cfg = config or DownloadConfig()
        self.source = source or self._default_source()
        self.converter = ImageConverter(self.cfg)

    # ---- สร้าง source จาก config ----------------------------------------
    def _default_source(self) -> ChapterSource:
        name = self.cfg.source
        try:
            if name == "generic-json":
                return build("source", name, config=self.cfg,
                             list_file=self.cfg.image_urls_file or None)
            if name == "generic-html":
                return build("source", name, config=self.cfg,
                             source_regex=self.cfg.source_regex or None)
            return build("source", name, config=self.cfg)
        except KeyError as exc:
            raise ValueError(
                f"ไม่รู้จัก source '{name}'. ที่รองรับ: {', '.join(available('source'))}"
            ) from exc

    # ---- ตั้งชื่อไฟล์ ----------------------------------------------------
    def _target_path(self, index: int) -> Path:
        out_dir = Path(self.cfg.out_dir or "orig")
        num = self.cfg.start_index + index - 1
        return out_dir / f"{self.cfg.prefix}-{num:04d}.webp"

    # ---- ดาวน์โหลด 1 รูป --------------------------------------------------
    def _download_one(self, ref: ImageRef) -> tuple[str, int, Optional[str]]:
        """คืน (dest_path, size, error)"""
        dest = self._target_path(ref.index)

        if dest.exists() and not self.cfg.overwrite:
            return str(dest), dest.stat().st_size, None

        headers = {"User-Agent": self.cfg.user_agent}
        if self.cfg.referer:
            headers["Referer"] = self.cfg.referer

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.cfg.retries + 1):
            try:
                with requests.get(ref.url, headers=headers, timeout=self.cfg.timeout,
                                  stream=True) as r:
                    r.raise_for_status()
                    chunks = bytearray()
                    for chunk in r.iter_content(chunk_size=self.cfg.chunk_size):
                        if chunk:
                            chunks.extend(chunk)
                # บังคับแปลงเป็น WebP แล้วบันทึก
                size = self.converter.save_webp(bytes(chunks), dest)
                return str(dest), size, None
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("ดาวน์โหลด %s ล้มเหลว (ครั้งที่ %d): %s", ref.url, attempt, exc)
        return str(dest), 0, f"{type(last_exc).__name__}: {last_exc}"

    # ---- ดาวน์โหลดทั้ง chapter ------------------------------------------
    def download_chapter(self, chapter_url: str) -> DownloadReport:
        refs = self.source.list_images(chapter_url)
        if self.cfg.limit and self.cfg.limit > 0:
            log.info("จำกัดการดาวน์โหลดไว้ที่ %d หน้าแรก (limit)", self.cfg.limit)
            refs = refs[: self.cfg.limit]
        report = DownloadReport(chapter_url=chapter_url, source=self.source.name,
                                out_dir=str(self.cfg.out_dir or "orig"), total=len(refs))
        if not refs:
            log.error("ไม่พบรูปใน chapter นี้")
            return report

        log.info("เริ่มดาวน์โหลด %d หน้า พร้อมกัน %d เธรด...",
                 len(refs), self.cfg.max_workers)

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.max_workers)) as pool:
            futures = {pool.submit(self._download_one, ref): ref for ref in refs}
            for fut in as_completed(futures):
                ref = futures[fut]
                dest, size, err = fut.result()
                if err:
                    report.failed += 1
                    report.errors[ref.url] = err
                elif size == 0 and not Path(dest).exists():
                    report.skipped += 1
                else:
                    report.downloaded += 1
                    report.files.append(dest)

        report.files.sort()
        self._write_manifest(report, refs)
        log.info(report.summary())
        return report

    # ---- เขียน manifest --------------------------------------------------
    def _manifest_file(self) -> Path:
        if self.cfg.manifest_path:
            return Path(self.cfg.manifest_path)
        return Path(".scanlation") / "download_manifest.json"

    def _write_manifest(self, report: DownloadReport, refs: List[ImageRef]) -> None:
        try:
            p = self._manifest_file()
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "chapter_url": report.chapter_url,
                "source": report.source,
                "out_dir": report.out_dir,
                "prefix": self.cfg.prefix,
                "start_index": self.cfg.start_index,
                "force_webp": self.cfg.force_webp,
                "webp_quality": self.cfg.webp_quality,
                "count": report.total,
                "downloaded": report.downloaded,
                "failed": report.failed,
                "pages": [
                    {
                        "index": i,
                        "file": f"{self.cfg.prefix}-{self.cfg.start_index + i - 1:04d}.webp",
                        "source_url": ref.url,
                        "page_label": ref.page_label,
                    }
                    for i, ref in enumerate(refs, start=1)
                ],
                "errors": report.errors,
            }
            p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001 - การเขียน manifest ไม่ควรทำให้ล้ม
            log.debug("เขียน manifest ไม่สำเร็จ:\n%s", traceback.format_exc())
