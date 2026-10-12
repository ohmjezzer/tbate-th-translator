# -*- coding: utf-8 -*-
"""ตั้งค่า logger มาตรฐานกลางของโปรเจกต์"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "scanlation", level: int = logging.INFO) -> logging.Logger:
    """คืน logger ที่ตั้งค่า format เรียบร้อยแล้ว (ตั้งค่าครั้งเดียว)"""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger("scanlation")
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True
    return logging.getLogger(name)
