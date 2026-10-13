# -*- coding: utf-8 -*-
"""
web
===
หน้า preview/แก้ไขผลการแปลแบบเบา ๆ (รันในเครื่อง)

    python -m scanlation.cli preview            # เปิดที่ http://127.0.0.1:8000
    python -m scanlation.cli preview --port 8080

API:
    GET  /                        หน้าเว็บ
    GET  /api/project             รายการหน้า (มี ready/orig/cleaned/out)
    GET  /api/page/<idx>          ข้อมูลหน้า (บอลลูน + คำแปล + media urls)
    POST /api/page/<idx>          บันทึกคำแปล + ค่าที่แก้ (meta)
    POST /api/page/<idx>/render   เรนเดอร์ใหม่ (ใช้ภาพ inpaint เป็นฐาน)
    POST /api/page/<idx>/prepare  ประมวลผลหน้าจากหน้าเว็บ (Detection→...→Typeset)
    GET  /media/<kind>/<idx>      orig | cleaned | out

ดูรายละเอียดที่ scanlation/web/app.py
"""

from .app import create_app, run_server

__all__ = ["create_app", "run_server"]
