# -*- coding: utf-8 -*-
"""
downloader/__init__.py
======================
สเตจ Download & Ingest: ดึงรูปจากเว็บ → บังคับแปลง WebP → วางใน input_dir
"""

from __future__ import annotations

from .base import ChapterSource, ImageRef
from .converter import ImageConverter
from .downloader import ChapterDownloader, DownloadReport
from .generic import GenericHtmlSource, GenericJsonSource
from .vortexscans import VortexScansSource

# นำเข้าคลาสด้านบนเพื่อให้ @register(...) ทำงานตอน import แพ็กเกจ
__all__ = [
    "ChapterSource",
    "ImageRef",
    "ImageConverter",
    "ChapterDownloader",
    "DownloadReport",
    "VortexScansSource",
    "GenericJsonSource",
    "GenericHtmlSource",
]
