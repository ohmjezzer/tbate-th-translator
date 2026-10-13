# -*- coding: utf-8 -*-
"""
detection/sam_segmenter.py
==========================
หัวใจของสเตจ Detection & Masking

แนวคิด "2 ชั้น" (สองคลาสสืบทอดกัน):
    BoxPromptSegmenter
        - จัดกลุ่ม TextBlock ที่ใกล้กัน -> "ผู้ต้องสงสัยว่าเป็นบอลลูน" 1 ก้อน
        - สร้าง mask (เริ่มจากสี่เหลี่ยมมุมขอบบอลลูน)
        - จำแนกประเภท speech / narration / sfx ด้วย heuristic
        - ไม่พึ่ง ML ใด ๆ -> ใช้เป็น fallback และใช้ unit test ได้

    SAMSegmenter(BoxPromptSegmenter)
        - override เฉพาะเมธอด _predict_mask()
        - ใช้ Segment Anything Model (Meta) ตัด mask ที่ "แนบเนียนไปตามรูปทรงจริง"
          ของบอลลูน (วงรี ขอบหยัก หางบอลลูน) แทนสี่เหลี่ยมแข็ง ๆ

ทำไม SAM จึงดีกว่าเดิม:
    โค้ดเดิมใช้ cv2.inpaint บน "สี่เหลี่ยม" รอบข้อความ -> โดนตัวอักษรศิลป์/เส้นขอบ
    บอลลูนลบไปด้วย ทำให้พื้นหลังเละ
    SAM ให้ mask ที่ตรงกับ "เนื้อในบอลลูน" จริง จึงลบเฉพาะตัวอักษรและเก็บเส้นขอบ
    บอลลูนไว้ครบ
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ..config import DetectionConfig
from ..schemas import BBox, BlockKind, Bubble, TextBlock
from ..utils import geometry as geo
from ..utils.logging_utils import get_logger
from ..utils.registry import register
from .base import BubbleSegmenter

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# คลาสฐาน: จัดกลุ่ม + จำแนกประเภท + mask เริ่มต้นแบบสี่เหลี่ยม
# ---------------------------------------------------------------------------
@register("segmenter", "box")
class BoxPromptSegmenter(BubbleSegmenter):
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.cfg = config or DetectionConfig()

    # -- 1) จัดกลุ่มบรรทัดข้อความให้เป็นบอลลูน -----------------------------
    def _group(self, blocks: List[TextBlock]) -> List[tuple[BBox, List[TextBlock]]]:
        """
        รวม TextBlock ที่มีช่องว่างระหว่างกันน้อยกว่า merge_gap ให้เป็นกลุ่มเดียว
        (ในมังงะ บรรทัดในบอลลูนเดียวกันจะห่างกันไม่เกินความสูงบรรทัด)
        """
        pairs = [(b.bbox, b) for b in blocks]
        gap_groups = geo.merge_overlapping(pairs, gap=self.cfg.merge_gap, threshold=0.0)

        # merge_overlapping คืน (BBox, [payload]) — payload อาจเป็น BBox หรือ TextBlock
        # ขึ้นกับว่าเราส่ง object อะไรเข้าไป; ที่นี่เราส่ง TextBlock จึงได้ TextBlock กลับ
        result: List[tuple[BBox, List[TextBlock]]] = []
        for merged_box, members in gap_groups:
            tblocks = [m for m in members if isinstance(m, TextBlock)]
            result.append((merged_box, tblocks))
        return result

    # -- 2) สร้าง mask (คลาสนี้ใช้สี่เหลี่ยม; SAM จะ override) -------------
    def _predict_mask(self, image: np.ndarray, box: BBox) -> np.ndarray:
        """mask เริ่มต้น: เต็มกรอบบอลลูน (สี่เหลี่ยม)"""
        h, w = image.shape[:2]
        m = np.zeros((h, w), dtype=bool)
        x0, y0, x1, y1 = box.clip(w, h).to_int()
        m[y0:y1, x0:x1] = True
        return m

    # -- 3) จำแนกประเภทกล่อง ---------------------------------------------
    def _classify(self, image: np.ndarray, box: BBox, mask: np.ndarray) -> BlockKind:
        """
        Heuristic:
            - วัดความสว่างเฉลี่ย *ภายใน mask* → ถ้าไม่ขาวพอ = SFX (ตัวอักษรบนภาพ)
            - ตรวจสีที่ "มุมทั้ง 4 ของกรอบ mask":
                * มุมเป็นภาพ (ไม่ขาว) → ทรงกลม/วงรี → SPEECH
                * มุมเป็นสีขาวเหมือนกัน → ทรงสี่เหลี่ยม → NARRATION
        หมายเหตุ: heuristic นี้แม่นเมื่อ mask มาจาก SAM/SAM2 (ตามรูปทรงจริง)
        """
        if mask.sum() == 0:
            return BlockKind.SFX
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]

        # ความสว่างเฉลี่ยภายใน mask
        inside_lum = float(gray[mask].mean())
        if inside_lum < 255 * self.cfg.sam_bubble_whiteness:
            return BlockKind.SFX

        # กรอบ (bounding box) ของ mask เอง
        ys, xs = np.where(mask)
        mx0, mx1, my0, my1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        bw, bh = mx1 - mx0, my1 - my0
        if bw < 4 or bh < 4:
            return BlockKind.SPEECH

        # สุ่มตรวจที่มุม 4 มุมของกรอบ mask
        c = max(2, min(bw, bh) // 8)
        dark = 255 * self.cfg.sam_bubble_whiteness * 0.8
        bright = 0
        for (py, px) in ((my0, mx0), (my0, mx1 - c), (my1 - c, mx0), (my1 - c, mx1 - c)):
            patch = gray[max(0, py):py + c, max(0, px):px + c]
            if patch.size and patch.mean() >= dark:
                bright += 1
        bright_frac = bright / 4.0

        # มุมสว่างเกือบทั้งหมด = กล่องสี่เหลี่ยมบรรยาย
        if bright_frac >= 0.75:
            return BlockKind.NARRATION
        return BlockKind.SPEECH

    # -- 4) เมธอดหลักของ interface ----------------------------------------
    def segment(self, image: np.ndarray, text_blocks: List[TextBlock]) -> List[Bubble]:
        h, w = image.shape[:2]
        bubbles: List[Bubble] = []
        for i, (gbox, members) in enumerate(self._group(text_blocks)):
            prompt_box = gbox.expand(self.cfg.sam_expand).clip(w, h)
            mask = self._predict_mask(image, prompt_box)
            if mask.sum() == 0:
                continue
            mask = geo.largest_component(mask)  # เก็บเฉพาะก้อนที่ใหญ่สุด
            kind = self._classify(image, gbox, mask)
            bubbles.append(
                Bubble(
                    uid=f"bub{i}",
                    bbox=gbox,
                    kind=kind,
                    mask=mask,
                    polygon=geo.polygon_from_mask(mask),
                    blocks=members,
                )
            )
        return bubbles


# ---------------------------------------------------------------------------
# SAM (Segment Anything Model) — Meta
# ---------------------------------------------------------------------------
@register("segmenter", "sam")
class SAMSegmenter(BoxPromptSegmenter):
    """
    ใช้ SAM ตัด mask ของบอลลูนแบบแม่นยำ (prompt = กรอบข้อความ)

    การใช้งาน:
        seg = SAMSegmenter(config)
        seg.segment(image, text_blocks)   # โหลดโมเดลครั้งแรกอัตโนมัติ

    ต้องมี:
        pip install git+https://github.com/facebookresearch/segment-anything.git
        ดาวน์โหลด checkpoint (sam_vit_h_4b8939.pth) ไปวางที่ cfg.sam_checkpoint

    หมายเหตุการขยายผล:
        - SAM2 (facebook/sam2) หรือ MobileSAM ใช้ interface เดียวกันได้
          เพียงสลับคลาส _load_model / _raw_predict (ดูตัวอย่างที่ inject ได้ผ่าน
          พารามิเตอร์ model_handle ด้านล่าง)
    """

    def __init__(self, config: Optional[DetectionConfig] = None, model_handle=None):
        super().__init__(config)
        self._model = model_handle       # รองรับ DI เพื่อทดสอบ
        self._predictor = None

    # ---- โหลดโมเดลแบบ lazy ----------------------------------------------
    @staticmethod
    def resolve_checkpoint(configured: str) -> Optional[str]:
        """
        หาไฟล์ checkpoint ของ SAM ที่ "มีอยู่จริง":
          1) ถ้า path ที่ตั้งไว้มีไฟล์ -> ใช้ตัวนั้น
          2) ถ้าไม่มี -> ค้นหาไฟล์ sam_*.pth ในโฟลเดอร์เดียวกัน (เช่นผู้ใช้โหลด vit_b
             แต่ตั้งค่าเป็น vit_h) -> ใช้ตัวที่เจอ (variant อาจต่างจากที่ตั้ง)
          3) ไม่เจอเลย -> คืน None

        คืนค่า: path (str) ของ checkpoint ที่ใช้ได้ หรือ None
        """
        p = Path(configured)
        if p.is_file():
            return str(p)
        # ค้นหา sam_*.pth ในโฟลเดอร์ของมัน (ถ้ามี) และใน weights/
        for folder in {p.parent, Path("weights")}:
            if folder and folder.is_dir():
                found = sorted(folder.glob("sam_*.pth"))
                if found:
                    return str(found[0])
        return None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:  # pragma: no cover - ขึ้นกับ environment
            raise RuntimeError(
                "ยังไม่ได้ติดตั้ง segment-anything (SAM) — จึงยังใช้ตัวตัดขอบเขตแบบ SAM ไม่ได้\n"
                "  เลือกทางใดทางหนึ่ง:\n"
                "    1) ติดตั้ง SAM (คุณภาพดีสุด):\n"
                "         pip install -r requirements-ml.txt\n"
                "         python scripts/download_weights.py --model vit_b\n"
                "    2) หรือใช้ตัวที่ไม่ต้องติดตั้งเพิ่ม (รันได้ทันที):\n"
                "         สร้าง .env แล้วใส่  SCANLATION_SEGMENTER=box\n"
                "         (หรือ fastsam ถ้าติดตั้ง ultralytics แล้ว)"
            ) from exc

        mtype = self.cfg.sam_model_type
        if mtype not in sam_model_registry:  # pragma: no cover
            raise ValueError(f"ไม่รู้จัก sam_model_type '{mtype}' (ต้องเป็น vit_h/vit_l/vit_b)")

        # --- ตรวจว่าไฟล์ checkpoint มีจริงไหม (ก่อนพยายามโหลด) ---
        ckpt = self.resolve_checkpoint(self.cfg.sam_checkpoint)
        if ckpt is None:
            raise RuntimeError(
                f"ไม่พบไฟล์โมเดล SAM: '{self.cfg.sam_checkpoint}'\n"
                f"  ดาวน์โหลด checkpoint ด้วยคำสั่ง:\n"
                f"      python scripts/download_weights.py --model {mtype}\n"
                f"  (หรือใช้ตัวที่ไม่ต้องมี checkpoint: SCANLATION_SEGMENTER=box)"
            )
        if ckpt != self.cfg.sam_checkpoint:
            log.warning("ไม่พบ '%s' แต่พบ '%s' — ใช้ไฟล์ที่พบแทน",
                        self.cfg.sam_checkpoint, ckpt)

        log.info("กำลังโหลด SAM (%s) จาก %s ...", mtype, ckpt)
        sam = sam_model_registry[mtype](checkpoint=ckpt)
        sam.to(device=self.cfg.sam_device)
        self._model = sam
        return sam

    def _ensure_predictor(self, image: np.ndarray):
        """สร้าง predictor และ set_image (ต้องทำใหม่ทุกครั้งที่เปลี่ยนภาพ)"""
        from segment_anything import SamPredictor

        if self._predictor is None:
            self._predictor = SamPredictor(self._load_model())
        # set_image แพง (encode ภาพ) จึงเรียกครั้งเดียวต่อภาพ
        self._predictor.set_image(image)
        return self._predictor

    # ---- override: ใช้ SAM ทำนาย mask จาก box prompt ---------------------
    def _predict_mask(self, image: np.ndarray, box: BBox) -> np.ndarray:
        predictor = self._ensure_predictor(image)
        x0, y0, x1, y1 = box.to_int()
        # box prompt: รูปแบบเดียวกับ SAM คือ [x0,y0,x1,y1]
        masks, scores, _ = predictor.predict(
            box=np.array([x0, y0, x1, y1]),
            multimask_output=True,   # คืน 3 mask ให้เลือก
        )
        # เลือก mask ที่คะแนนดีสุด แต่ต้อง "ครอบข้อความในกรอบ" ด้วย
        best_i, best_score = 0, -1.0
        for i, (m, s) in enumerate(zip(masks, scores)):
            # ให้โบนัสกับ mask ที่กินพื้นที่กรอบมาก (มักเป็นบอลลูนจริง)
            fill = m[y0:y1, x0:x1].mean() if m[y0:y1, x0:x1].size else 0.0
            score = float(s) + 0.25 * float(fill)
            if score > best_score:
                best_i, best_score = i, score

        mask = masks[best_i].astype(bool)
        # --- post-processing: อุดรู + ขยายเล็กน้อย + เก็บก้อนใหญ่สุด ---
        mask = geo.largest_component(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        return mask


# ---------------------------------------------------------------------------
# FastSAM — ทางเลือกที่เบาและเร็ว (รันบน CPU ได้)
# ---------------------------------------------------------------------------
@register("segmenter", "fastsam")
class FastSAMSegmenter(BoxPromptSegmenter):
    """
    ทางเลือกแทน SAM แบบดั้งเดิม เมื่อไม่มี GPU หรือต้องการความเร็ว

        pip install ultralytics
        (ดาวน์โหลด FastSAM-x.pt อัตโนมัติเมื่อเรียกครั้งแรก)

    FastSAM เป็น YOLO-based → เร็วมาก แต่ mask ละเอียดน้อยกว่า SAM เล็กน้อย
    เหมาะกับบอลลูนคำพูดที่มีเส้นขอบชัด
    """

    def __init__(self, config: Optional[DetectionConfig] = None,
                 weights: str = "FastSAM-x.pt", device: Optional[str] = None):
        super().__init__(config)
        self.weights = weights
        self.device = device or self.cfg.sam_device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import FastSAM  # import ตอนใช้

            log.info("กำลังโหลด FastSAM (%s)...", self.weights)
            self._model = FastSAM(self.weights)
        return self._model

    def _predict_mask(self, image: np.ndarray, box: BBox) -> np.ndarray:
        model = self._ensure_model()
        x0, y0, x1, y1 = box.to_int()
        # FastSAM รับ bbox prompt ผ่านพารามิเตอร์ bboxes (พิกัดในกรอบภาพ)
        results = model(image, bboxes=[[x0, y0, x1, y1]], device=self.device, verbose=False)
        res = results[0]
        h, w = image.shape[:2]
        if res.masks is None or len(res.masks.data) == 0:
            return super()._predict_mask(image, box)
        # รวม mask ทุกอันจาก prompt เดียว แล้ว resize ให้เท่าภาพต้นฉบับ
        m = res.masks.data.cpu().numpy()
        combined = (m.sum(axis=0) > 0).astype(np.uint8)
        combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        return geo.largest_component(combined)
