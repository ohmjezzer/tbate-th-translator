# -*- coding: utf-8 -*-
"""
registry.py
===========
Simple plugin registry — หัวใจของความ "ขยายผลได้" (Scalability)

แนวคิด: ตัว pipeline ไม่รู้จักชื่อคลาสจริง ๆ แต่ขอ object ผ่าน registry ตาม
"ชื่อสั้น" ที่ระบุใน config เช่น config.inpainting.method = "lama"

เมื่อต้องการเพิ่มโมเดลใหม่ (เช่น MAT, MI-GAN, SD-inpaint) เพียง:
    1. เขียนคลาสที่สืบทอดจาก interface เดิม
    2. @register("inpaint", "mat") ครอบคลุม
    3. ตั้งค่าใน config — ไม่ต้องแก้ pipeline เลย
"""

from __future__ import annotations

from typing import Any, Callable, Dict

_REGISTRY: Dict[str, Dict[str, Callable[..., Any]]] = {}


def register(kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator สำหรับลงทะเบียนคลาส/ฟังก์ชันเข้ากับ registry"""

    def deco(obj: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY.setdefault(kind, {})[name] = obj
        return obj

    return deco


def build(kind: str, name: str, **kwargs: Any) -> Any:
    """สร้างอินสแตนซ์จากชื่อที่ลงทะเบียนไว้ พร้อมส่ง kwargs ให้ constructor"""
    if kind not in _REGISTRY or name not in _REGISTRY[kind]:
        available = ", ".join(sorted(_REGISTRY.get(kind, {}))) or "(ว่าง)"
        raise KeyError(f"ไม่พบ '{name}' ในหมวด '{kind}'. ที่มี: {available}")
    return _REGISTRY[kind][name](**kwargs)


def available(kind: str) -> list[str]:
    """ดูรายชื่อที่ลงทะเบียนไว้ในหมวดนั้น ๆ"""
    return sorted(_REGISTRY.get(kind, {}))


class Registry:
    """Wrapper แบบ object สำหรับผู้ที่ชอบเรียก registry.build(...)"""

    build = staticmethod(build)
    available = staticmethod(available)
    register = staticmethod(register)
