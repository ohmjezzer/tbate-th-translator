# -*- coding: utf-8 -*-
"""
monitoring/ — ระบบเฝ้าสังเกต (Monitoring) ทุกสเตจของ pipeline

    from .metrics import MetricsCollector, PageMetrics

ระบบนี้ทำหน้าที่:
    1. รวบรวมตัวชี้วัด (metrics) ของแต่ละสเตจ (Detection/Translation/Inpainting/Typesetting)
    2. สร้างภาพวินิจฉัย (debug overlays) เทียบดูต้นฉบับ-มาสก์-ผลลัพธ์
    3. สรุปประเด็นที่น่าสงสัย (issues) เป็นรหัสมาตรฐานพร้อมข้อความอธิบาย
    4. เขียนไฟล์รายงาน (JSON) และภาพ PNG ลง .scanlation/monitoring/
"""

from .metrics import MetricsCollector, PageMetrics

__all__ = ["MetricsCollector", "PageMetrics"]
