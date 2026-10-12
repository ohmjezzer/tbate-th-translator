# -*- coding: utf-8 -*-
"""
web/app.py
==========
ระบบ Preview & Editor (รันในเครื่อง) สำหรับตรวจ/แก้ผลการแปลก่อนส่งงาน

    python -m scanlation.cli preview [--host 127.0.0.1] [--port 8000]

ฟีเจอร์:
    - ดูผลการแปลเทียบกับต้นฉบับ (สลับ orig / cleaned / out ได้)
    - แก้ข้อความคำแปลในหน้าเว็บได้ทันที
    - ลากย้ายข้อความ (offset) และย่อ/ขยายกรอบข้อความ (bbox) ให้ตรงบอลลูน
    - ปรับขนาดฟอนต์ (font_size), การจัดชิด (align), สีตัวอักษร (color) ต่อบอลลูน
    - กด "บันทึก & เรนเดอร์ใหม่" เพื่อให้ Typesetter วาดใหม่ตามที่แก้

ผลลัพธ์ที่บันทึก:
    - แก้ไขเก็บใน meta ของแต่ละ Bubble แล้วเขียนกลับไฟล์
      .scanlation/3_translation_pXXXX.json
    - การเรนเดอร์ใหม่ใช้ภาพที่ inpaint แล้ว (.scanlation/2_cleaned_pXXXX.png)
      เป็นฐาน -> เขียนทับ out/page-XXXX.png

ใช้ Flask (ดู requirements-web.txt)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ScanlationConfig
from ..schemas import Page
from ..utils import image_io
from ..utils.logging_utils import get_logger

log = get_logger(__name__)

_IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg", ".bmp")


# ---------------------------------------------------------------------------
# ค้นหาไฟล์ภาพของแต่ละหน้าจากเลขหน้า (page-0001.*)
# ---------------------------------------------------------------------------
def _find_file(directory: Path, index: int, exts=_IMAGE_EXTS) -> Optional[Path]:
    if not directory.exists():
        return None
    for p in sorted(directory.glob(f"*{index:04d}*")):
        if p.suffix.lower() in exts:
            return p
    return None


class PreviewServer:
    """รวมตรรกะการอ่าน/บันทึก/เรนเดอร์ ที่ frontend เรียกใช้ผ่าน HTTP API"""

    def __init__(self, cfg: Optional[ScanlationConfig] = None):
        self.cfg = cfg or ScanlationConfig()
        self.input_dir = Path(self.cfg.paths.input_dir)
        self.output_dir = Path(self.cfg.paths.output_dir)
        self.work_dir = Path(self.cfg.paths.work_dir)

    # ---- รายการหน้า -------------------------------------------------------
    def _page_indexes(self) -> List[int]:
        idxs: set[int] = set()
        for d in (self.input_dir, self.output_dir):
            if d.exists():
                for p in d.iterdir():
                    m = re.search(r"(\d+)", p.stem)
                    if p.suffix.lower() in _IMAGE_EXTS and m:
                        idxs.add(int(m.group(1)))
        for d in (self.work_dir,):
            if d.exists():
                for p in d.glob("*_p*.json"):
                    m = re.search(r"_p(\d+)\.json$", p.name)
                    if m:
                        idxs.add(int(m.group(1)))
        return sorted(idxs)

    def project(self) -> Dict[str, Any]:
        pages = []
        for i in self._page_indexes():
            pages.append({
                "index": i,
                "orig": _find_file(self.input_dir, i) is not None,
                "out": _find_file(self.output_dir, i) is not None,
                "cleaned": (self.work_dir / f"2_cleaned_p{i:04d}.png").exists(),
            })
        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "work_dir": str(self.work_dir),
            "pages": pages,
        }

    # ---- โหลด/บันทึกหน้า -------------------------------------------------
    def _page_json(self, index: int) -> Optional[Path]:
        for name in (f"3_translation_p{index:04d}.json", f"1_detection_p{index:04d}.json"):
            p = self.work_dir / name
            if p.exists():
                return p
        return None

    def get_page(self, index: int) -> Dict[str, Any]:
        p = self._page_json(index)
        if p is None:
            raise FileNotFoundError(
                f"ไม่พบไฟล์ข้อมูลหน้าที่ {index} ใน {self.work_dir}/ — "
                f"รัน `scanlation.cli run {index}` ก่อน"
            )
        page = Page.load_json(p)

        # ขนาดภาพ (จาก out -> cleaned -> orig)
        width, height = page.width, page.height
        src = _find_file(self.output_dir, index) or (self.work_dir / f"2_cleaned_p{index:04d}.png")
        if src and src.exists():
            w, h = image_io.load_pil(src).size
            width, height = w, h

        bubbles = []
        for b in page.bubbles:
            bubbles.append({
                "uid": b.uid,
                "bbox": [b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1],
                "kind": b.kind.value,
                "source_text": b.source_text,
                "translation": b.translation,
                "meta": {k: v for k, v in (b.meta or {}).items() if k != "reserved"},
                "has_mask": b.mask is not None,
            })
        return {
            "index": index,
            "width": width, "height": height,
            "media": self._media_urls(index),
            "bubbles": bubbles,
        }

    def save_page(self, index: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """บันทึกคำแปล + ค่าที่แก้ (จาก frontend) กลับลงไฟล์ JSON"""
        p = self._page_json(index)
        if p is None:
            raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูลหน้าที่ {index}")
        page = Page.load_json(p)

        edits = {b["uid"]: b for b in payload.get("bubbles", []) if "uid" in b}
        for bub in page.bubbles:
            e = edits.get(bub.uid)
            if not e:
                continue
            if "translation" in e:
                bub.set_translation(str(e["translation"]))
            # meta: รวมค่าเดิม + ค่าที่แก้ แล้วลบค่าที่ตั้งเป็น null (กลับเป็น auto)
            meta = dict(bub.meta or {})
            for k, v in (e.get("meta") or {}).items():
                if v is None:
                    meta.pop(k, None)
                else:
                    meta[k] = v
            bub.meta = meta

        page.save_json(p, include_mask=True)
        return {"ok": True, "saved": str(p)}

    # ---- เรนเดอร์ใหม่ -----------------------------------------------------
    def render_page(self, index: int) -> Dict[str, Any]:
        """วาดข้อความตามคำแปล/ค่าที่แก้ แล้วเขียนทับ out/page-XXXX.png"""
        p = self._page_json(index)
        if p is None:
            raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูลหน้าที่ {index}")
        page = Page.load_json(p)

        # ใช้ภาพที่ inpaint แล้วเป็นฐาน (ถ้าไม่มี ใช้ต้นฉบับแทน พร้อมเตือน)
        base_path = self.work_dir / f"2_cleaned_p{index:04d}.png"
        if not base_path.exists():
            base_path = _find_file(self.input_dir, index)  # type: ignore[assignment]
            log.warning("ไม่พบภาพ inpaint ของหน้า %d — ใช้ต้นฉบับแทน", index)
        if base_path is None or not base_path.exists():
            raise FileNotFoundError(f"ไม่พบภาพฐานของหน้าที่ {index}")

        image = image_io.load_image_rgb(base_path)
        from ..typesetting import Typesetter

        ts = Typesetter(self.cfg.typesetting)
        out = ts.render(image, page)

        out_path = self.output_dir / f"page-{index:04d}.png"
        image_io.save_image(out, out_path)
        return {"ok": True, "out": str(out_path), "media": self._media_urls(index)}

    # ---- URL ของสื่อ ------------------------------------------------------
    def _media_urls(self, index: int) -> Dict[str, Optional[str]]:
        def url(kind: str, path: Optional[Path]) -> Optional[str]:
            return f"/media/{kind}/{index}" if (path and path.exists()) else None

        return {
            "orig": url("orig", _find_file(self.input_dir, index)),
            "cleaned": url("cleaned", self.work_dir / f"2_cleaned_p{index:04d}.png"),
            "out": url("out", _find_file(self.output_dir, index)),
        }

    def media_path(self, kind: str, index: int) -> Optional[Path]:
        if kind == "orig":
            return _find_file(self.input_dir, index)
        if kind == "cleaned":
            p = self.work_dir / f"2_cleaned_p{index:04d}.png"
            return p if p.exists() else None
        if kind == "out":
            return _find_file(self.output_dir, index)
        return None


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
def create_app(cfg: Optional[ScanlationConfig] = None):
    """สร้าง Flask app (import flask แบบ lazy)"""
    try:
        from flask import Flask, jsonify, request, send_file, send_from_directory
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ต้องติดตั้ง Flask สำหรับโหมด preview: pip install -r requirements-web.txt"
        ) from exc

    server = PreviewServer(cfg)
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/api/project")
    def api_project():
        return jsonify(server.project())

    @app.get("/api/page/<int:index>")
    def api_get_page(index: int):
        try:
            return jsonify(server.get_page(index))
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.post("/api/page/<int:index>")
    def api_save_page(index: int):
        try:
            return jsonify(server.save_page(index, request.get_json(force=True) or {}))
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.post("/api/page/<int:index>/render")
    def api_render_page(index: int):
        try:
            return jsonify(server.render_page(index))
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # noqa: BLE001
            log.exception("เรนเดอร์หน้า %d ล้มเหลว", index)
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.get("/media/<kind>/<int:index>")
    def api_media(kind: str, index: int):
        path = server.media_path(kind, index)
        if path is None:
            return jsonify({"error": "ไม่พบไฟล์"}), 404
        resp = send_file(str(path))
        resp.headers["Cache-Control"] = "no-store"    # ให้เห็นผลหลังเรนเดอร์ใหม่ทันที
        return resp

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000,
               cfg: Optional[ScanlationConfig] = None, debug: bool = False) -> None:
    app = create_app(cfg)
    print(f"\n  🖼  Scanlation Preview  →  http://{host}:{port}\n"
          f"     (กด Ctrl+C เพื่อหยุด)\n")
    app.run(host=host, port=port, debug=debug, threaded=True)
