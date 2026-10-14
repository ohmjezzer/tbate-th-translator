# -*- coding: utf-8 -*-
"""
monitoring/metrics.py
=====================
MetricsCollector — ตัวรวบรวมตัวชี้วัด (metrics) ของ pipeline ต่อ 1 หน้า

แนวคิดการออกแบบ (Design rationale):
    - แต่ละสเตจ (Detection / Translation / Inpainting / Typesetting) จะมี
      "ตัวชี้วัดที่ตอบคำถามหลัก" ของตัวเอง เช่น
        - Detection:   จับข้อความได้กี่บล็อก/บอลลูน? มั่นใจต่ำไหม? กรอบบวมเกินไปไหม?
        - Translation: แปลครบหรือยัง? มีไทยปนอังกฤษ (leftover English) ตรงไหน?
        - Inpainting:  ลบพื้นหลังโดนอะไรบ้าง? ครอบคลุมพื้นที่ภาพกี่ %?
        - Typesetting: วาดทับสำเร็จครบไหม?
    - ทุก metrics ถูกเก็บเป็น dict เรียบ ๆ แล้ว "แปลงเป็น issues" (รหัสมาตรฐาน)
      อัตโนมัติ ผู้ใช้อ่านภาพรวมได้ทันทีจากรหัส + ข้อความ (ไม่ต้องไข log)
    - ทุก metrics/issues ถูกเขียนลงไฟล์ JSON (และภาพ PNG debug) ใน .scanlation/monitoring/
      เพื่อให้ตรวจย้อนหลัง/ส่งต่อให้ frontend แสดงผลได้

ข้อมูลที่เก็บต่อ 1 หน้า (PageMetrics):
    page_index, image_path, stages {detection/translation/inpainting/typesetting},
    issues[], timings_ms{}, artifacts{}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..schemas import BlockKind, Page
from ..utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class PageMetrics:
    """โครงสร้างผล metrics ของ 1 หน้า"""

    page_index: int
    image_path: str = ""
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    issues: List[Dict[str, str]] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path ของภาพ debug

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """
    รวบรวมและวิเคราะห์ metrics ของ pipeline

    วิธีใช้:
        collector = MetricsCollector(work_dir)
        m = collector.begin_page(index=1, image_path="orig/page-0001.webp")
        collector.analyze_detection(m, page)
        collector.analyze_translation(m, page)
        collector.analyze_inpainting(m, page, mask)
        collector.analyze_typesetting(m, page)
        collector.finalize_page(m, page=page, mask=mask, image=img)
    """

    # เกณฑ์เตือนมาตรฐาน (ปรับได้)
    LOW_CONF_THRESHOLD = 0.50      # ค่า OCR confidence ที่ถือว่า "ต่ำ"
    MAX_BUBBLE_AREA_RATIO = 0.60   # บอลลูนหนึ่งก้อนไม่ควรกินพื้นที่เกิน 60% ของภาพ
    MAX_MASK_COVERAGE = 0.35       # mask รวมไม่ควรเกิน 35% ของภาพ (เกิน = อาจลบพื้นหลังเยอะผิดปกติ)

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.metrics_dir = self.work_dir / "monitoring"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # เริ่มเก็บ metrics ของ 1 หน้า
    # ------------------------------------------------------------------
    def begin_page(self, index: int, image_path: str = "") -> PageMetrics:
        return PageMetrics(page_index=index, image_path=image_path)

    # ------------------------------------------------------------------
    # วิเคราะห์ผลสเตจ Detection
    # ------------------------------------------------------------------
    def analyze_detection(self, m: PageMetrics, page: Page) -> None:
        blocks = [b for bub in page.bubbles for b in bub.blocks]
        total_blocks = len(blocks)
        total_bubbles = len(page.bubbles)

        low_conf = [b for b in blocks if b.confidence < self.LOW_CONF_THRESHOLD]
        kinds: Dict[str, int] = {}
        for b in page.bubbles:
            kinds[b.kind.value] = kinds.get(b.kind.value, 0) + 1

        oversized = []
        page_area = max(1, page.width * page.height)
        for bub in page.bubbles:
            ratio = bub.area_px / page_area
            if ratio > self.MAX_BUBBLE_AREA_RATIO:
                oversized.append((bub.uid, round(ratio * 100, 1)))

        m.stages["detection"] = {
            "text_blocks": total_blocks,
            "bubbles": total_bubbles,
            "low_confidence_blocks": len(low_conf),
            "bubble_kinds": kinds,
            "oversized_bubbles": [{"uid": u, "area_pct": a} for u, a in oversized],
        }

        if total_blocks == 0:
            m.issues.append({
                "code": "NO_TEXT_DETECTED",
                "message": "OCR ไม่พบข้อความใด ๆ เลย — ภาพอาจไม่มีตัวอักษร/สแกนเบลอ/ผิด format",
            })
        if low_conf:
            m.issues.append({
                "code": "LOW_CONFIDENCE",
                "message": f"มี {len(low_conf)} บล็อกที่ OCR มั่นใจต่ำ (<{self.LOW_CONF_THRESHOLD:.0%}) — อาจจับผิด/จับไม่ครบ",
            })
        if oversized:
            m.issues.append({
                "code": "OVERSIZED_BUBBLE",
                "message": f"มีบอลลูนขนาดใหญ่ผิดปกติ ({', '.join(u for u, _ in oversized)}) — อาจรวมกรอบผิด/กินพื้นหลังเกิน",
            })

    # ------------------------------------------------------------------
    # วิเคราะห์ผลสเตจ Translation
    # ------------------------------------------------------------------
    def analyze_translation(self, m: PageMetrics, page: Page,
                            allowed_latin: Optional[List[str]] = None) -> None:
        total = len(page.bubbles)
        translated = 0
        missing = []
        leftover = []

        allowed = [a for a in (allowed_latin or []) if a]
        for bub in page.bubbles:
            th = bub.translation.strip()
            if th:
                translated += 1
                if self._has_leftover_english(th, allowed):
                    leftover.append(bub.uid)
            else:
                missing.append(bub.uid)

        m.stages["translation"] = {
            "bubbles_total": total,
            "translated": translated,
            "missing": len(missing),
            "leftover_english": len(leftover),
            "missing_uids": missing,
            "leftover_uids": leftover,
        }

        if missing:
            m.issues.append({
                "code": "MISSING_TRANSLATION",
                "message": f"บอลลูน {len(missing)}/{total} ยังไม่มีคำแปล ({', '.join(missing[:5])}{'...' if len(missing) > 5 else ''})",
            })
        if leftover:
            m.issues.append({
                "code": "LEFTOVER_ENGLISH",
                "message": f"บอลลูน {len(leftover)}/{total} มีภาษาอังกฤษหลงเหลือในคำแปล ({', '.join(leftover[:5])}{'...' if len(leftover) > 5 else ''})",
            })

    @staticmethod
    def _has_leftover_english(text: str, allowed_latin: Optional[List[str]] = None) -> bool:
        """
        ตรวจว่ามี "ภาษาอังกฤษหลงเหลือ" ในคำแปลไทยหรือไม่
        (ตัดคำที่อยู่ใน allowed_latin — เช่นชื่อเฉพาะที่ glossary อนุญาต — ออกก่อน)
        """
        import re

        cleaned = text
        for w in (allowed_latin or []):
            if w:
                cleaned = cleaned.replace(w, " ")
        candidates = re.findall(r"[A-Za-z]{3,}", cleaned)
        return bool(candidates)

    # ------------------------------------------------------------------
    # วิเคราะห์ผลสเตจ Inpainting (ต้องมี mask)
    # ------------------------------------------------------------------
    def analyze_inpainting(self, m: PageMetrics, page: Page, mask: Optional[np.ndarray]) -> None:
        if mask is None:
            m.stages["inpainting"] = {"status": "skipped", "reason": "no mask"}
            return

        total_px = mask.size
        masked_px = int((mask > 0).sum())
        coverage = masked_px / total_px if total_px else 0.0

        m.stages["inpainting"] = {
            "mask_pixels": masked_px,
            "coverage_ratio": round(coverage, 4),
            "coverage_pct": round(coverage * 100, 2),
        }

        if masked_px == 0:
            m.issues.append({
                "code": "EMPTY_MASK",
                "message": "Mask ว่างเปล่า — ไม่มีพื้นที่ถูกลบ (อาจเพราะ detection ไม่เจออะไรเลย หรือ mask สร้างผิดพลาด)",
            })
        elif coverage > self.MAX_MASK_COVERAGE:
            m.issues.append({
                "code": "HIGH_MASK_COVERAGE",
                "message": f"Mask ครอบคลุมพื้นที่ภาพ {coverage * 100:.1f}% — สูงผิดปกติ อาจลบพื้นหลัง/ฉากหลังเกินจริง",
            })

    # ------------------------------------------------------------------
    # วิเคราะห์ผลสเตจ Typesetting (ผลลัพธ์สุดท้าย)
    # ------------------------------------------------------------------
    def analyze_typesetting(self, m: PageMetrics, page: Page) -> None:
        total = len(page.bubbles)
        rendered = sum(1 for bub in page.bubbles if bub.translation.strip())
        m.stages["typesetting"] = {
            "bubbles_total": total,
            "rendered": rendered,
            "unrendered": total - rendered,
        }
        if rendered < total:
            m.issues.append({
                "code": "UNRENDERED_BUBBLES",
                "message": f"มีบอลลูน {total - rendered}/{total} ที่ยังไม่มีคำแปลจึงไม่ได้วาดทับ",
            })

    # ------------------------------------------------------------------
    # จบการเก็บของ 1 หน้า: รวมผล + เขียนไฟล์
    # ------------------------------------------------------------------
    def finalize_page(self, m: PageMetrics, page: Page,
                      mask: Optional[np.ndarray] = None,
                      image: Optional[np.ndarray] = None) -> PageMetrics:
        """เรียกหลังประมวลผลครบทุกสเตจแล้ว เพื่อรวมผลและเขียนไฟล์"""
        # วิเคราะห์ตามสเตจ (เรียกเฉพาะตัวที่ยังไม่มีข้อมูล เพื่อให้ override ได้)
        if "detection" not in m.stages:
            self.analyze_detection(m, page)
        if "translation" not in m.stages:
            self.analyze_translation(m, page)
        if "inpainting" not in m.stages:
            self.analyze_inpainting(m, page, mask)
        if "typesetting" not in m.stages:
            self.analyze_typesetting(m, page)

        # สร้างภาพ debug ก่อนเขียน JSON เพื่อให้ artifacts ถูกบันทึกด้วย
        if mask is not None:
            try:
                stem = f"page-{m.page_index:04d}"
                mask_path = self.metrics_dir / f"{stem}-mask.png"
                self._save_mask_image(mask, mask_path)
                m.artifacts["mask"] = mask_path.name
                # overlay ต้องมีภาพต้นฉบับด้วย (อาจไม่มีในกรณีทดสอบบางส่วน)
                if image is not None:
                    overlay_path = self.metrics_dir / f"{stem}-overlay.png"
                    self._save_mask_overlay(image, mask, page, overlay_path)
                    m.artifacts["overlay"] = overlay_path.name
            except Exception as exc:  # noqa: BLE001
                log.warning("สร้างภาพ debug หน้า %d ไม่สำเร็จ: %s", m.page_index, exc)

        # เขียน JSON (หลัง artifacts เติมครบแล้ว)
        out_json = self.metrics_dir / f"page-{m.page_index:04d}.json"
        out_json.write_text(json.dumps(m.to_dict(), ensure_ascii=False, indent=2),
                            encoding="utf-8")
        log.info("เขียน metrics หน้า %d -> %s (%d issues)", m.page_index, out_json, len(m.issues))
        return m

    # ------------------------------------------------------------------
    # ภาพ debug
    # ------------------------------------------------------------------
    @staticmethod
    def _save_mask_image(mask: np.ndarray, out_path: Path) -> None:
        """บันทึก binary mask เป็นภาพขาวดำ (ขาว = จุดที่ถูกลบ)"""
        import cv2

        img = ((mask > 0).astype(np.uint8)) * 255
        cv2.imwrite(str(out_path), img)

    @staticmethod
    def _save_mask_overlay(image: np.ndarray, mask: np.ndarray, page: Page,
                           out_path: Path) -> None:
        """วาดภาพต้นฉบับ + ทับสีแดงบริเวณ mask + กรอบบอลลูน (ใช้ตรวจว่าลบถูกจุดไหม)"""
        import cv2

        img = image.copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        # ทับ mask ด้วยสีแดงโปร่งแสง
        red = np.zeros_like(img)
        red[..., 0] = 255
        alpha = 0.35
        mask_bool = mask > 0
        img[mask_bool] = (img[mask_bool] * (1 - alpha) + red[mask_bool] * alpha).astype(np.uint8)

        # วาดกรอบบอลลูน (สีตามประเภท)
        colors = {
            BlockKind.SPEECH: (0, 200, 0),        # เขียว
            BlockKind.NARRATION: (255, 140, 0),   # ส้ม
            BlockKind.SFX: (220, 0, 220),         # ม่วงแดง
            BlockKind.CREDIT: (0, 200, 200),      # เหลืองอมเขียว
            BlockKind.UNKNOWN: (140, 140, 140),
        }
        for bub in page.bubbles:
            x0, y0, x1, y1 = bub.bbox.to_int()
            color = colors.get(bub.kind, (255, 255, 255))
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            cv2.putText(img, bub.uid, (x0, max(12, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # ------------------------------------------------------------------
    # อ่าน metrics ที่บันทึกไว้ (ใช้โดย web API)
    # ------------------------------------------------------------------
    def load_page(self, index: int) -> Optional[PageMetrics]:
        d = self.load(self.metrics_dir, index)
        if d is None:
            return None
        m = PageMetrics(page_index=d.get("page_index", index),
                        image_path=d.get("image_path", ""))
        m.stages = d.get("stages", {})
        m.issues = d.get("issues", [])
        m.timings_ms = d.get("timings_ms", {})
        m.artifacts = d.get("artifacts", {})
        return m

    @staticmethod
    def load(metrics_dir: str | Path, index: int) -> Optional[Dict[str, Any]]:
        p = Path(metrics_dir) / f"page-{index:04d}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def load_all(metrics_dir: str | Path) -> List[Dict[str, Any]]:
        out = []
        for p in sorted(Path(metrics_dir).glob("page-*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return out

    def summary(self) -> Dict[str, Any]:
        """สรุปภาพรวมทุกหน้า: จำนวนหน้า, หน้าที่มี issues, สถิติรวมต่อสเตจ"""
        pages = self.load_all(self.metrics_dir)
        total = len(pages)
        issue_count = sum(len(p.get("issues", [])) for p in pages)
        pages_with_issues = sum(1 for p in pages if p.get("issues"))

        # นับรหัส issue ที่พบบ่อย
        code_freq: Dict[str, int] = {}
        for p in pages:
            for it in p.get("issues", []):
                code = it.get("code", "UNKNOWN")
                code_freq[code] = code_freq.get(code, 0) + 1

        return {
            "total_pages": total,
            "pages_with_issues": pages_with_issues,
            "total_issues": issue_count,
            "issue_code_frequency": code_freq,
            "pages": [
                {
                    "page_index": p.get("page_index"),
                    "issues": p.get("issues", []),
                    "stages": {
                        k: {kk: vv for kk, vv in v.items() if not kk.endswith("_uids")}
                        for k, v in p.get("stages", {}).items()
                    },
                }
                for p in pages
            ],
        }
