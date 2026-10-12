# -*- coding: utf-8 -*-
"""
web
===
หน้า preview/แก้ไขผลการแปลแบบเบา ๆ (รันในเครื่อง)

    python -m scanlation.cli preview            # เปิดที่ http://127.0.0.1:8000
    python -m scanlation.cli preview --port 8080

ดูรายละเอียดที่ scanlation/web/app.py
"""

from .app import create_app, run_server

__all__ = ["create_app", "run_server"]
