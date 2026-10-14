# -*- coding: utf-8 -*-
"""
web/app.py
==========
ระบบ Preview & Editor (รันในเครื่อง) สำหรับตรวจ/แก้ผลการแปลก่อนส่งงาน

    python -m scanlation.cli preview [--host 127.0.0.1] [--port 8000]

ฟีเจอร์:
    - ดูผลการแปลเทียบกับต้นฉบับ (สลับ orig / cleaned / out ได้)
    - แก้ข้อความคำแปล, ลากย้าย/ย่อขยายบอลลูน, ปรับฟอนต์/สี/จัดชิด
    - **เพิ่ม / ลบบอลลูน** ได้เอง (คลิกวาดกรอบใหม่ หรือกดลบ)
    - **แปรงลบพื้นที่เอง (mask brush) → ให้ AI inpaint** ลบข้อความ/ลายน้ำที่หลุด
    - **Undo** การ inpaint ย้อนกลับได้
    - กด "บันทึก & เรนเดอร์ใหม่" เพื่อให้ Typesetter วาดใหม่ตามที่แก้

ผลลัพธ์ที่บันทึก:
    - แก้ไขเก็บใน meta ของแต่ละ Bubble แล้วเขียนกลับ .scanlation/3_translation_pXXXX.json
    - การเรนเดอร์ใหม่ใช้ภาพที่ inpaint แล้ว (.scanlation/2_cleaned_pXXXX.png) เป็นฐาน
      -> เขียนทับ out/page-XXXX.png
    - การแปรงลบจะอัปเดต 2_cleaned_pXXXX.png (สำรอง .prev.png ไว้ให้ undo)

API:
    GET    /                        หน้าเว็บ
    GET    /api/project             รายการหน้า
    GET    /api/page/<idx>          ข้อมูลหน้า
    POST   /api/page/<idx>          บันทึกคำแปล + meta
    POST   /api/page/<idx>/render   เรนเดอร์ใหม่
    POST   /api/page/<idx>/bubbles  เพิ่ม/ลบบอลลูน (op=add|delete)
    POST   /api/page/<idx>/inpaint  แปรงลบ + AI inpaint (op=inpaint|undo)
    GET    /media/<kind>/<idx>      orig | cleaned | out

ใช้ Flask (ดู requirements-web.txt)
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import ScanlationConfig
from ..inpainting import build_text_mask
from ..utils.image_compressor import (
    compress_image_file,
    ext_for,
    human_size,
    normalize_format,
    target_bytes_from_mb,
)
from ..schemas import BBox, BlockKind, Bubble, Page, TextBlock
from ..utils import image_io
from ..utils.logging_utils import get_logger
from ..monitoring import MetricsCollector

log = get_logger(__name__)

_IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg", ".bmp")


# ---------------------------------------------------------------------------
# ข้อยกเว้นเฉพาะของฝั่ง preview (เพื่อคืนข้อความช่วยเหลือให้ frontend)
# ---------------------------------------------------------------------------
class PageNotReady(Exception):
    """หน้าที่ขอ ยังไม่มีข้อมูลใน .scanlation/ (ยังไม่ได้ประมวลผล)"""

    def __init__(self, index: int, command: Optional[str] = None, message: str = ""):
        self.index = index
        self.command = command
        self.message = message or f"หน้านี้ยังไม่ได้ประมวลผล (ไม่มีข้อมูลใน .scanlation/)"
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {"error": self.message, "needs_run": True,
                "index": self.index, "command": self.command}


class PrepareError(Exception):
    """การประมวลผลหน้า (Detection→Translate→Inpaint→Typeset) ล้มเหลว"""


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

    def run_command(self, index: int) -> str:
        """คำสั่ง CLI ที่ผู้ใช้ควรรันเพื่อประมวลผลหน้านี้"""
        return f"python -m scanlation.cli run {index}"

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
                # ready = มีข้อมูลข้อความ/คำแปลใน .scanlation แล้ว (เปิดแก้ไขได้)
                "ready": self._page_json(i) is not None,
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
            raise PageNotReady(index, self.run_command(index))
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
            "output": self._output_info(index),
            "bubbles": bubbles,
        }

    def _output_info(self, index: int) -> Dict[str, Any]:
        """ข้อมูลไฟล์ผลลัพธ์ (ขนาดจริง + เป้าหมายที่ตั้งไว้) ให้ UI แสดงผล"""
        o = self.cfg.output
        target = target_bytes_from_mb(getattr(o, "target_size_mb", 1.0))
        f = _find_file(self.output_dir, index)
        info: Dict[str, Any] = {
            "exists": f is not None,
            "target_mb": float(getattr(o, "target_size_mb", 1.0)),
            "format": str(getattr(o, "format", "webp")),
            "compress": bool(getattr(o, "compress", True)),
        }
        if f is not None and f.exists():
            size = f.stat().st_size
            info.update({"path": f.name, "size_bytes": size,
                         "size_human": human_size(size),
                         "under_target": size <= target})
        return info

    def save_page(self, index: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """บันทึกคำแปล + ค่าที่แก้ (จาก frontend) กลับลงไฟล์ JSON"""
        p = self._page_json(index)
        if p is None:
            raise PageNotReady(index, self.run_command(index))
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

    # ---- เพิ่ม / ลบบอลลูนด้วยมือ -----------------------------------------
    def add_bubble(self, index: int, bbox: List[float], kind: str = "speech",
                   translation: str = "", source_text: str = "",
                   erase: bool = False) -> Dict[str, Any]:
        """
        เพิ่มบอลลูนใหม่ (วาดกรอบเองในหน้าเว็บ)

        erase=True -> ลบข้อความอังกฤษเดิมในกรอบนี้ด้วย AI ทันที (ถ้ามี)
        (ช่วยกรณี detection ตกหล่น: ผู้ใช้เพิ่มกรอบเองแล้วให้ระบบลบข้อความเดิมให้)
        """
        p = self._page_json(index)
        if p is None:
            raise PageNotReady(index, self.run_command(index))
        page = Page.load_json(p)

        # uid ใหม่: bub-manual-<n> (หาเลขที่ไม่ซ้ำ)
        existing = {b.uid for b in page.bubbles}
        n = 1
        while f"bub-manual-{n}" in existing:
            n += 1
        uid = f"bub-manual-{n}"
        box = BBox.from_list([float(v) for v in bbox])
        try:
            k = BlockKind(kind)
        except ValueError:
            k = BlockKind.SPEECH

        bubble = Bubble(uid=uid, bbox=box, kind=k,
                        blocks=[TextBlock(uid=f"{uid}-t0", bbox=box, source_text=source_text)])
        bubble.set_translation(translation or "")
        page.bubbles.append(bubble)
        page.save_json(p, include_mask=True)

        result: Dict[str, Any] = {"ok": True, "uid": uid, "saved": str(p)}
        if erase:
            # ลบข้อความเดิมที่ยังค้างในกรอบ (โหมด text-only: เก็บพื้นหลังไว้)
            try:
                erased = self.inpaint_box(index, box.as_list(), text_only=True)
                result["erased"] = erased.get("ok", False)
            except Exception as exc:  # noqa: BLE001 - การลบไม่ควรทำให้การเพิ่มล้ม
                log.warning("ลบข้อความในกรอบ %s ไม่สำเร็จ: %s", uid, exc)
                result["erased"] = False
                result["erase_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def delete_bubble(self, index: int, uid: str) -> Dict[str, Any]:
        """ลบบอลลูนออกจากหน้า"""
        p = self._page_json(index)
        if p is None:
            raise PageNotReady(index, self.run_command(index))
        page = Page.load_json(p)
        before = len(page.bubbles)
        page.bubbles = [b for b in page.bubbles if b.uid != uid]
        if len(page.bubbles) == before:
            return {"ok": False, "error": f"ไม่พบบอลลูน uid={uid}"}
        page.save_json(p, include_mask=True)
        return {"ok": True, "saved": str(p), "remaining": len(page.bubbles)}

    # ---- แปรงลบพื้นที่เอง + AI inpaint ------------------------------------
    def _cleaned_path(self, index: int) -> Path:
        return self.work_dir / f"2_cleaned_p{index:04d}.png"

    def _base_image_path(self, index: int) -> Optional[Path]:
        """ภาพฐานสำหรับ inpaint: ใช้ cleaned ถ้ามี ไม่งั้นใช้ต้นฉบับ"""
        c = self._cleaned_path(index)
        if c.exists():
            return c
        return _find_file(self.input_dir, index)

    def _strokes_to_mask(self, strokes: List[Dict], shape: Tuple[int, int]) -> np.ndarray:
        """
        แปลง stroke จาก frontend เป็น binary mask (uint8 0/255) ขนาดเท่าภาพ

        strokes: [{"points": [[x,y],...], "radius": r}, ...]
        """
        import cv2

        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for s in strokes or []:
            pts = s.get("points") or []
            radius = int(s.get("radius", 18))
            for pt in pts:
                try:
                    x, y = int(pt[0]), int(pt[1])
                except (TypeError, ValueError, IndexError):
                    continue
                cv2.circle(mask, (x, y), max(1, radius), 255, -1)
            # เชื่อมจุดที่ติดกันให้เป็นเส้นต่อเนื่อง
            if len(pts) >= 2:
                for a, b in zip(pts, pts[1:]):
                    try:
                        cv2.line(mask, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                                 255, max(1, radius * 2))
                    except (TypeError, ValueError, IndexError):
                        continue
        return mask

    def _apply_inpaint_mask(self, index: int, mask: np.ndarray, grow: int = 4,
                            method: Optional[str] = None,
                            note: str = "") -> Dict[str, Any]:
        """
        แกนกลางของการลบพื้นที่: รับ mask (พิกเซลภาพ) → inpaint → เขียนทับ cleaned
        แล้วเรนเดอร์ใหม่ (สำรอง .prev.png ไว้ให้ undo)

        ทั้งการแปรงลบ (strokes) และการลบในกรอบบอลลูน (box) จะเรียกเมธอดนี้ร่วมกัน
        เพื่อให้พฤติกรรมเหมือนกันเป๊ะ (และแก้ที่เดียว)
        """
        import cv2

        base_path = self._base_image_path(index)
        if base_path is None or not base_path.exists():
            raise PageNotReady(index, self.run_command(index),
                               f"ไม่พบภาพของหน้าที่ {index} (ต้องมีใน orig/ ก่อน)")

        image = image_io.load_image_rgb(base_path)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        if int((mask > 0).sum()) == 0:
            return {"ok": False, "error": "ไม่มีพื้นที่ให้ลบ (mask ว่าง)"}

        if grow > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
            mask = cv2.dilate(mask, k, iterations=1)

        # เลือก inpainter
        from ..pipeline import ScanlationPipeline
        cfg = self.cfg
        if method:
            cfg.inpainting.method = method
        method_name = ScanlationPipeline.resolve_inpainter_method(
            cfg.inpainting.method, allow_fallback=getattr(cfg.inpainting, "auto_fallback", True))
        inpainter = ScanlationPipeline(cfg).inpainter
        log.info("[preview] inpaint หน้า %d ด้วย '%s' (%d px%s)",
                 index, method_name, int((mask > 0).sum()), f" — {note}" if note else "")

        cleaned = inpainter.inpaint(image, mask)

        # สำรองภาพเดิมไว้ให้ undo (ครั้งล่าสุด)
        dest = self._cleaned_path(index)
        dest.parent.mkdir(parents=True, exist_ok=True)
        prev = dest.with_suffix(".prev.png")
        if dest.exists():
            prev.write_bytes(dest.read_bytes())
        elif base_path.resolve() != dest.resolve():
            prev.write_bytes(Path(base_path).read_bytes())

        image_io.save_image(cleaned, dest)

        # เรนเดอร์ผลลัพธ์ใหม่ทันที เพื่อให้เห็นภาพรวม
        try:
            self.render_page(index)
        except Exception as exc:  # noqa: BLE001
            log.warning("เรนเดอร์หลัง inpaint ไม่สำเร็จ: %s", exc)

        return {"ok": True, "method": method_name,
                "erased_px": int((mask > 0).sum()),
                "media": self._media_urls(index)}

    def inpaint_region(self, index: int, strokes: List[Dict], grow: int = 4,
                       method: Optional[str] = None) -> Dict[str, Any]:
        """
        ลบพื้นที่ที่ผู้ใช้แปรงคลุม ด้วยโมเดล inpaint (LaMa/OpenCV/Median ตาม config)
        แล้วเขียนทับภาพ cleaned (สำรอง .prev.png ไว้ให้ undo)

        method: ระบุชื่อวิธีใน InpaintingConfig.method (ถ้าไม่ระบุใช้ตาม config) — 'auto' ได้
        """
        base_path = self._base_image_path(index)
        if base_path is None or not base_path.exists():
            raise PageNotReady(index, self.run_command(index),
                               f"ไม่พบภาพของหน้าที่ {index} (ต้องมีใน orig/ ก่อน)")
        h, w = image_io.load_image_rgb(base_path).shape[:2]
        mask = self._strokes_to_mask(strokes, (h, w))
        return self._apply_inpaint_mask(index, mask, grow=grow, method=method,
                                        note=f"แปรง {len(strokes)} เส้น")

    def inpaint_box(self, index: int, bbox: List[float], grow: int = 4,
                    method: Optional[str] = None, text_only: bool = True,
                    fallback_full: bool = True) -> Dict[str, Any]:
        """
        ลบ "ข้อความที่ตกค้าง" ภายในกรอบที่กำหนด (ใช้เมื่อ detection จับไม่ครบ
        แล้วผู้ใช้เพิ่มบอลลูนเอง หรือวาดกรอบเอง)

        text_only=True → ลบเฉพาะพิกเซลที่เป็นตัวอักษร (เก็บพื้นหลัง)
        fallback_full=True → ถ้าหา "ตัวอักษร" ในกรอบไม่เจอเลย ให้ลบทั้งกรอบแทน
        """
        import cv2

        base_path = self._base_image_path(index)
        if base_path is None or not base_path.exists():
            raise PageNotReady(index, self.run_command(index),
                               f"ไม่พบภาพของหน้าที่ {index} (ต้องมีใน orig/ ก่อน)")
        image = image_io.load_image_rgb(base_path)
        h, w = image.shape[:2]

        x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
        x0, x1 = sorted((max(0, min(x0, w)), max(0, min(x1, w))))
        y0, y1 = sorted((max(0, min(y0, h)), max(0, min(y1, h))))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return {"ok": False, "error": "กรอบเล็กเกินไป"}

        mask = np.zeros((h, w), dtype=np.uint8)
        if text_only:
            # ใช้ตรรกะสร้าง mask จาก "บรรทัดข้อความ" ชุดเดียวกับสเตจ Inpainting
            try:
                from ..schemas import BlockKind, Bubble, TextBlock
                box = BBox(x0, y0, x1, y1)
                tmp = Bubble(uid="box-erase", bbox=box, kind=BlockKind.UNKNOWN,
                             blocks=[TextBlock(uid="box-erase-t0", bbox=box)])
                mask = build_text_mask(image, [tmp], self.cfg.inpainting)
            except Exception as exc:  # noqa: BLE001
                log.warning("สร้าง text-mask ในกรอบไม่สำเร็จ (%s) — ใช้ลบทั้งกรอบแทน", exc)
                mask = np.zeros((h, w), dtype=np.uint8)
        if int((mask > 0).sum()) == 0 and fallback_full:
            mask[y0:y1, x0:x1] = 255

        return self._apply_inpaint_mask(index, mask, grow=grow, method=method,
                                        note=f"กรอบ [{x0},{y0},{x1},{y1}]")

    def undo_inpaint(self, index: int) -> Dict[str, Any]:
        """ย้อนการ inpaint ครั้งล่าสุด (คืนภาพ cleaned จาก .prev.png)"""
        dest = self._cleaned_path(index)
        prev = dest.with_suffix(".prev.png")
        if not prev.exists():
            return {"ok": False, "error": "ไม่มีประวัติให้ย้อนกลับ (undo)"}
        # สลับ: เก็บภาพปัจจุบันไว้เป็น .prev แต่ใหม่ เพื่อให้กด undo/redo สลับได้
        if dest.exists():
            tmp = dest.read_bytes()
            dest.write_bytes(prev.read_bytes())
            prev.write_bytes(tmp)
        else:
            dest.write_bytes(prev.read_bytes())
        try:
            self.render_page(index)
        except Exception as exc:  # noqa: BLE001
            log.warning("เรนเดอร์หลัง undo ไม่สำเร็จ: %s", exc)
        return {"ok": True, "media": self._media_urls(index)}

    # ---- เรนเดอร์ใหม่ -----------------------------------------------------
    def _cleanup_stale_outputs(self, index: int, keep: Path) -> None:
        """ลบไฟล์ผลลัพธ์เดิมของหน้าเดียวกันที่มีนามสกุลอื่น (กันไฟล์เก่าค้าง)"""
        for f in self.output_dir.glob(f"*{index:04d}*"):
            if f.suffix.lower() in _IMAGE_EXTS and f.resolve() != keep.resolve():
                try:
                    f.unlink()
                    log.info("[preview] ลบไฟล์ผลลัพธ์เดิมทิ้ง: %s", f.name)
                except OSError:
                    pass

    def render_page(self, index: int, target_mb: Optional[float] = None,
                    fmt: Optional[str] = None, compress: Optional[bool] = None
                    ) -> Dict[str, Any]:
        """
        วาดข้อความตามคำแปล/ค่าที่แก้ แล้วเขียนทับไฟล์ผลลัพธ์ใน out/

        target_mb / fmt / compress : override ค่าจาก config เฉพาะครั้งนี้ (จาก UI)
            - target_mb = ขนาดสูงสุดต่อไฟล์ (MB)
            - fmt       = "webp" | "jpeg" | "png" | "keep"
            - compress  = False เพื่อไม่บีบอัดเลย
        """
        p = self._page_json(index)
        if p is None:
            raise PageNotReady(index, self.run_command(index))
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

        # --- กำหนดฟอร์แมต/นามสกุลผลลัพธ์ ---
        o = self.cfg.output
        raw_fmt = str(fmt if fmt is not None else getattr(o, "format", "webp")).strip().lower()
        ext = ".png" if raw_fmt == "keep" else ext_for(raw_fmt)
        out_path = self.output_dir / f"page-{index:04d}{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image_io.save_image(out, out_path)

        # --- บีบอัดตามขนาดเป้าหมาย (ถ้าเปิด) ---
        info: Dict[str, Any] = {"compressed": False}
        do_compress = bool(getattr(o, "compress", True)) if compress is None else bool(compress)
        if do_compress:
            tgt_mb = float(target_mb if target_mb is not None
                           else getattr(o, "target_size_mb", 1.0))
            use_fmt = (out_path.suffix.lstrip(".") or "webp") if raw_fmt == "keep" \
                else normalize_format(raw_fmt)
            try:
                res = compress_image_file(
                    out_path, target_bytes_from_mb(tgt_mb), fmt=use_fmt,
                    min_quality=int(getattr(o, "min_quality", 40)),
                    max_quality=int(getattr(o, "max_quality", 92)),
                    allow_downscale=bool(getattr(o, "allow_downscale", True)),
                    max_downscale_ratio=float(getattr(o, "max_downscale_ratio", 0.6)),
                )
                out_path = Path(res.path)
                info = {
                    "compressed": True,
                    "ok": res.ok,
                    "size_bytes": res.size_bytes,
                    "size_human": human_size(res.size_bytes),
                    "quality": res.quality,
                    "scale": res.scale,
                    "target_mb": tgt_mb,
                    "format": res.format,
                }
            except Exception as exc:  # noqa: BLE001 - บีบอัดล้มไม่ควรทำให้เรนเดอร์ล้ม
                log.warning("บีบอัดผลลัพธ์หน้า %d ไม่สำเร็จ: %s", index, exc)
                info = {"compressed": False, "error": f"{type(exc).__name__}: {exc}"}

        self._cleanup_stale_outputs(index, out_path)
        return {"ok": True, "out": str(out_path), "info": info,
                "media": self._media_urls(index)}

    # ---- ประมวลผลหน้าจาก UI (Detection→Translate→Inpaint→Typeset) --------
    def prepare_page(self, index: int, pipeline=None) -> Dict[str, Any]:
        """
        ประมวลผลหน้าที่ระบุ "จากหน้าเว็บ" ได้เลย (ไม่ต้องออกไปรัน CLI เอง)
        แล้วคืนข้อมูลหน้า (พร้อมให้แก้ไข) กลับไป

        pipeline: ฉีดเข้ามาได้เพื่อทดสอบ (ค่าเริ่มต้น = ScanlationPipeline ปกติ)
        """
        orig = _find_file(self.input_dir, index)
        if orig is None:
            raise PageNotReady(
                index, None,
                f"ไม่พบภาพต้นฉบับของหน้าที่ {index} ในโฟลเดอร์ '{self.input_dir}' "
                f"— วางไฟล์ หรือดาวน์โหลดก่อน: python -m scanlation.cli download <url>",
            )

        if pipeline is None:
            from ..pipeline import ScanlationPipeline

            pipeline = ScanlationPipeline(self.cfg)

        log.info("[preview] เริ่มประมวลผลหน้า %d: %s", index, orig.name)
        try:
            pipeline.process_page(orig, index=index, do_translate=True, save=True)
        except Exception as exc:  # noqa: BLE001 - ส่งข้อความจริงกลับให้ UI
            raise PrepareError(f"{type(exc).__name__}: {exc}") from exc

        log.info("[preview] ประมวลผลหน้า %d เสร็จ", index)
        return self.get_page(index)

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
        """คืน path ของสื่อ (เป็น absolute path เสมอ เพื่อให้ส่งไฟล์ได้ถูกต้อง)"""
        if kind == "orig":
            p = _find_file(self.input_dir, index)
        elif kind == "cleaned":
            cand = self.work_dir / f"2_cleaned_p{index:04d}.png"
            p = cand if cand.exists() else None
        elif kind == "out":
            p = _find_file(self.output_dir, index)
        else:
            p = None
        return p.resolve() if p is not None else None

    # ---- Monitoring ----------------------------------------------------------
    def page_metrics(self, index: int) -> Dict[str, Any]:
        """อ่าน metrics ของหน้า (ถ้ามีไฟล์)"""
        collector = MetricsCollector(self.work_dir)
        m = collector.load_page(index)
        if m is None:
            return {"error": f"ยังไม่มี metrics ของหน้าที่ {index} (อาจยังไม่ได้ประมวลผล)"}
        return m.to_dict()

    def metrics_summary(self) -> Dict[str, Any]:
        """สรุป metrics ทุกหน้า (เฉพาะหน้าที่มีไฟล์)"""
        collector = MetricsCollector(self.work_dir)
        return collector.summary()

    def monitoring_path(self, index: int, filename: str) -> Optional[Path]:
        """คืน path ของไฟล์ใน .scanlation/monitoring/ (ตรวจสอบชื่อไฟล์ก่อน)"""
        mon_dir = Path(self.work_dir) / "monitoring"
        if not mon_dir.exists():
            return None
        # ป้องกัน path traversal
        if ".." in filename or "/" in filename:
            return None
        cand = mon_dir / filename
        if not cand.exists() or not cand.name.startswith(f"page-{index:04d}-"):
            return None
        return cand.resolve()


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
        except PageNotReady as exc:
            return jsonify(exc.to_dict()), 404
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.post("/api/page/<int:index>")
    def api_save_page(index: int):
        try:
            return jsonify(server.save_page(index, request.get_json(force=True) or {}))
        except PageNotReady as exc:
            return jsonify(exc.to_dict()), 404
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.post("/api/page/<int:index>/render")
    def api_render_page(index: int):
        """เรนเดอร์ใหม่ + บีบอัด: รับ {"target_mb":1.0,"format":"webp","compress":true} ได้"""
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(server.render_page(
                index,
                target_mb=body.get("target_mb"),
                fmt=body.get("format"),
                compress=body.get("compress"),
            ))
        except PageNotReady as exc:
            return jsonify(exc.to_dict()), 404
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # noqa: BLE001
            log.exception("เรนเดอร์หน้า %d ล้มเหลว", index)
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.post("/api/page/<int:index>/bubbles")
    def api_bubbles(index: int):
        """เพิ่ม/ลบบอลลูนด้วยมือ: {"op":"add", bbox, kind, translation} | {"op":"delete", uid}"""
        body = request.get_json(force=True) or {}
        op = (body.get("op") or "add").lower()
        try:
            if op == "delete":
                return jsonify(server.delete_bubble(index, body.get("uid", "")))
            # add (default)
            bbox = body.get("bbox") or [0, 0, 0, 0]
            return jsonify(server.add_bubble(
                index, bbox, kind=body.get("kind", "speech"),
                translation=body.get("translation", ""),
                source_text=body.get("source_text", ""),
                erase=bool(body.get("erase", False)),
            ))
        except PageNotReady as exc:
            return jsonify(exc.to_dict()), 404
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # noqa: BLE001
            log.exception("จัดการบอลลูนหน้า %d ล้มเหลว", index)
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.post("/api/page/<int:index>/inpaint")
    def api_inpaint(index: int):
        """แปรงลบ + AI inpaint: {"op":"inpaint"|"undo", strokes, grow, method}"""
        body = request.get_json(force=True) or {}
        op = (body.get("op") or "inpaint").lower()
        try:
            if op == "undo":
                return jsonify(server.undo_inpaint(index))
            if op == "box":
                # ลบข้อความในกรอบ (ใช้เมื่อเพิ่มบอลลูนเอง/วาดกรอบเอง)
                return jsonify(server.inpaint_box(
                    index, body.get("bbox") or [0, 0, 0, 0],
                    grow=int(body.get("grow", 4)),
                    method=body.get("method") or None,
                    text_only=bool(body.get("text_only", True)),
                ))
            return jsonify(server.inpaint_region(
                index, body.get("strokes") or [],
                grow=int(body.get("grow", 4)),
                method=body.get("method") or None,
            ))
        except PageNotReady as exc:
            return jsonify(exc.to_dict()), 404
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # noqa: BLE001
            log.exception("inpaint หน้า %d ล้มเหลว", index)
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.post("/api/page/<int:index>/prepare")
    def api_prepare_page(index: int):
        """ประมวลผลหน้าจากหน้าเว็บ (Detection→Translate→Inpaint→Typeset)"""
        try:
            return jsonify(server.prepare_page(index))
        except PageNotReady as exc:
            return jsonify(exc.to_dict()), 404
        except PrepareError as exc:
            return jsonify({"error": str(exc),
                            "hint": "ตรวจการตั้งค่า/การติดตั้ง (เช่น SAM, LaMa, API key) "
                                    "แล้วลองใหม่ — หรือรันเอง: "
                                    f"python -m scanlation.cli run {index}"}), 500
        except Exception as exc:  # noqa: BLE001
            log.exception("ประมวลผลหน้า %d ล้มเหลว", index)
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.get("/api/output-settings")
    def api_output_settings():
        """ค่าตั้งการบันทึก/บีบอัดปัจจุบัน (ให้ UI ใช้เป็นค่าเริ่มต้น)"""
        o = server.cfg.output
        return jsonify({
            "compress": bool(getattr(o, "compress", True)),
            "target_mb": float(getattr(o, "target_size_mb", 1.0)),
            "format": str(getattr(o, "format", "webp")),
            "min_quality": int(getattr(o, "min_quality", 40)),
            "max_quality": int(getattr(o, "max_quality", 92)),
            "allow_downscale": bool(getattr(o, "allow_downscale", True)),
            "max_downscale_ratio": float(getattr(o, "max_downscale_ratio", 0.6)),
        })

    @app.get("/api/metrics")
    def api_metrics_summary():
        """สรุป metrics ทุกหน้า"""
        return jsonify(server.metrics_summary())

    @app.get("/api/metrics/<int:index>")
    def api_page_metrics(index: int):
        """metrics ของหน้าที่ระบุ"""
        data = server.page_metrics(index)
        if "error" in data:
            return jsonify(data), 404
        return jsonify(data)

    @app.get("/monitoring/<int:index>/<filename>")
    def api_monitoring_file(index: int, filename: str):
        """ส่งไฟล์ภาพวินิจฉัย (mask / overlay) ของหน้าที่ระบุ"""
        path = server.monitoring_path(index, filename)
        if path is None:
            return jsonify({"error": "ไม่พบไฟล์วินิจฉัย"}), 404
        try:
            resp = send_file(str(path))
        except FileNotFoundError:
            return jsonify({"error": "ไฟล์หายไปจากดิสก์"}), 404
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/media/<kind>/<int:index>")
    def api_media(kind: str, index: int):
        path = server.media_path(kind, index)
        if path is None:
            return jsonify({"error": "ไม่พบไฟล์"}), 404
        # ⚠️ ต้องส่ง "absolute path" ให้ send_file เสมอ
        # เพราะ Flask ตีความ relative path เทียบกับ app.root_path (โฟลเดอร์
        # ของแพ็กเกจ scanlation/web) ไม่ใช่ current working directory
        # ถ้าส่ง relative (เช่น "out/page-0001.png") จะกลายเป็น
        # scanlation/web/out/page-0001.png -> FileNotFoundError -> 500
        try:
            resp = send_file(str(Path(path).resolve()))
        except FileNotFoundError:
            return jsonify({"error": f"ไม่พบไฟล์จริงบนดิสก์: {path}"}), 404
        resp.headers["Cache-Control"] = "no-store"    # ให้เห็นผลหลังเรนเดอร์ใหม่ทันที
        return resp

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000,
               cfg: Optional[ScanlationConfig] = None, debug: bool = False) -> None:
    app = create_app(cfg)
    print(f"\n  🖼  Scanlation Preview  →  http://{host}:{port}\n"
          f"     (กด Ctrl+C เพื่อหยุด)\n")
    app.run(host=host, port=port, debug=debug, threaded=True)
