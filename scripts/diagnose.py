#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnose.py
===================
เครื่องมือ "วินิจฉัย" (diagnostic sandbox) — แยกดูทีละสเตจว่าส่วนไหนทำให้ผลไม่เนียน

    python scripts/diagnose.py <image>                 # วิเคราะห์ภาพเดียว
    python scripts/diagnose.py orig/page-0002.webp --out diag/p2
    python scripts/diagnose.py <image> --segmenter box --inpaint median   # บังคับ component
    python scripts/diagnose.py <image> --no-translate                     # ข้ามการแปล (ไม่ต้องใช้ API key)

ผลลัพธ์ในโฟลเดอร์ --out (ค่าเริ่มต้น diag/):
    0_original.png      ภาพต้นฉบับ
    1_ocr.png           กรอบข้อความที่ OCR อ่านได้ (แดง) + ข้อความ
    2_masks.png         mask ที่ segmenter ตัดได้ (ตามชนิด) + กรอบบอลลูน (ฟ้า)
    3_cleaned.png       ภาพหลัง inpaint (ลบข้อความเดิม)
    4_final.png         ภาพสุดท้าย (ข้อความไทยทับ) — ถ้ามีการแปล
    report.txt          สรุปจำนวน/ข้อความ/ค่า conf + คำเตือน (เช่น mask ไม่คลุมข้อความ)

ใช้ตอบคำถาม: "ความไม่เนียนเกิดจาก component ไหน"
    - กรอบ OCR ผิด/เล็กเกิน  -> Detection (OCR)
    - mask ไม่คลุมข้อความ     -> Detection (segmenter: SAM/box)
    - ภาพหลัง inpaint ยังมีตัวอักษร -> Inpainting (LaMA/median/opencv)
    - คำแปลมั่ว/ไม่ตรง          -> Translation (LLM) หรือ OCR อ่านผิด
    - ข้อความล้น/สีเพี้ยน        -> Typesetting (Pillow)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# ให้ import scanlation ได้เมื่อรันจาก root ของโปรเจกต์
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from scanlation.config import ScanlationConfig  # noqa: E402
from scanlation.env import apply_env, load_dotenv  # noqa: E402
from scanlation.schemas import BlockKind  # noqa: E402
from scanlation.utils import image_io  # noqa: E402

# สีตามชนิดกล่อง (RGB)
KIND_COLOR = {
    "speech": (66, 133, 244),
    "narration": (52, 168, 83),
    "sfx": (244, 160, 0),
    "credit": (150, 150, 150),
    "unknown": (200, 200, 200),
}


def _draw_boxes(img, boxes, color, labels=None, thickness=2):
    out = img.copy()
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = [int(v) for v in b]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
        if labels and i < len(labels) and labels[i]:
            t = str(labels[i])[:28]
            cv2.putText(out, t, (x0 + 2, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="วินิจฉัย pipeline แยกทีละสเตจ")
    ap.add_argument("image", help="path ของภาพที่ต้องการวินิจฉัย")
    ap.add_argument("--out", default="diag", help="โฟลเดอร์ผลลัพธ์ (ค่าเริ่มต้น diag/)")
    ap.add_argument("--segmenter", help="บังคับ segmenter: auto|sam|fastsam|box")
    ap.add_argument("--inpaint", help="บังคับ inpaint: lama|lama-torch|opencv|median")
    ap.add_argument("--ocr", help="บังคับ OCR engine: easyocr|tesseract")
    ap.add_argument("--no-translate", action="store_true", help="ข้ามสเตจแปล (ไม่ต้อง API key)")
    ap.add_argument("--config", help="ไฟล์ config.json")
    args = ap.parse_args()

    load_dotenv()
    cfg = ScanlationConfig.load(args.config) if args.config and Path(args.config).exists() \
        else apply_env(ScanlationConfig())
    if args.segmenter:
        cfg.detection.segmenter = args.segmenter
    if args.inpaint:
        cfg.inpainting.method = args.inpaint
    if args.ocr:
        cfg.detection.ocr_engine = args.ocr

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"✗ ไม่พบไฟล์ภาพ: {img_path}", file=sys.stderr)
        return 2
    image = image_io.load_image_rgb(img_path)
    h, w = image.shape[:2]
    image_io.save_image(image, out_dir / "0_original.png")
    print(f"ภาพ: {img_path}  ({w}x{h})")

    lines: list[str] = [f"DIAGNOSE: {img_path} ({w}x{h})",
                        f"config: segmenter={cfg.detection.segmenter} "
                        f"inpaint={cfg.inpainting.method} ocr={cfg.detection.ocr_engine} "
                        f"backend={cfg.translation.backend} model={cfg.translation.model}",
                        ""]

    from scanlation.pipeline import ScanlationPipeline

    # ---------------- (1) OCR ----------------
    from scanlation.detection.ocr_detector import EasyOCRDetector, TesseractDetector
    if cfg.detection.ocr_engine == "tesseract":
        detector = TesseractDetector(cfg.detection)
    else:
        detector = EasyOCRDetector(cfg.detection, gpu=cfg.detection.sam_device.startswith("cuda"))
    print("(1) OCR ...")
    text_blocks = detector.detect(image)
    ocr_boxes = [tb.bbox.as_list() for tb in text_blocks]
    ocr_labels = [f"{tb.source_text[:20]} ({tb.confidence:.2f})" for tb in text_blocks]
    image_io.save_image(_draw_boxes(image, ocr_boxes, (255, 60, 60), ocr_labels, 2),
                        out_dir / "1_ocr.png")
    lines.append(f"[1] OCR พบ {len(text_blocks)} บล็อก (engine={cfg.detection.ocr_engine})")
    for tb in text_blocks:
        lines.append(f"    {tb.bbox.to_int()} conf={tb.confidence:.2f} '{tb.source_text}'")

    # ---------------- (2) Segmentation / mask ----------------
    from scanlation.detection import DetectionPipeline
    from scanlation.detection.sam_segmenter import BoxPromptSegmenter, FastSAMSegmenter, SAMSegmenter
    seg_name = ScanlationPipeline.resolve_segmenter_name(
        cfg.detection.segmenter, allow_fallback=cfg.detection.auto_fallback)
    print(f"(2) Segmentation (ใช้ '{seg_name}') ...")
    seg_cls = {"sam": SAMSegmenter, "fastsam": FastSAMSegmenter, "box": BoxPromptSegmenter}[seg_name]
    pipeline = DetectionPipeline(detector, seg_cls(cfg.detection), cfg.detection)
    page = pipeline.detect_page(image, index=0, image_path=str(img_path))

    # วาด mask (ทับด้วยสีตามชนิด)
    overlay = image.copy().astype(np.float32)
    for bub in page.bubbles:
        if bub.mask is None:
            continue
        col = KIND_COLOR.get(bub.kind.value, (200, 200, 200))
        m = bub.mask
        overlay[m] = (overlay[m] * 0.45 + np.array(col) * 0.55)
    masks_vis = overlay.astype(np.uint8)
    masks_vis = _draw_boxes(masks_vis, [b.bbox.as_list() for b in page.bubbles],
                            (0, 180, 255), [b.kind.value for b in page.bubbles], 2)
    image_io.save_image(masks_vis, out_dir / "2_masks.png")
    lines.append("")
    lines.append(f"[2] Segmenter '{seg_name}' ได้ {len(page.bubbles)} บอลลูน")
    for bub in page.bubbles:
        cov = ""
        if bub.mask is not None:
            mask_pts = int(bub.mask.sum())
            # ใช้ "กรอบที่ขยายแล้ว" เป็นตัวหาร เพราะ mask สร้างจาก prompt box ที่ขยาย
            # (เดิมหารด้วย bbox ตรง ๆ ทำให้ได้ >100% แล้วอ่านสับสน)
            ref = bub.bbox.expand(cfg.detection.sam_expand).clip(w, h)
            ref_area = int(ref.area) or 1
            cov = f" mask={mask_pts}px ({mask_pts/ref_area*100:.0f}% ของกรอบ+expand)"
        lines.append(f"    {bub.uid} [{bub.kind.value}] box={bub.bbox.to_int()}{cov} "
                     f"'{bub.source_text[:30]}'")

    # ---------------- (3) Inpainting ----------------
    from scanlation.inpainting import build_bubble_mask, build_text_mask
    print("(3) Inpainting ...")
    text_mask = build_text_mask(image, page.bubbles, cfg.inpainting)
    image_io.save_image(np.stack([text_mask] * 3, axis=2), out_dir / "2b_text_mask.png")
    pipeline_i = ScanlationPipeline(cfg)
    inpainter = pipeline_i.inpainter
    inpaint_name = type(inpainter).__name__
    cleaned = inpainter.inpaint(image, text_mask)
    image_io.save_image(cleaned, out_dir / "3_cleaned.png")

    # ตรวจว่า mask คลุม "ข้อความที่ OCR เจอ" ครบไหม (ตัวชี้วัดสำคัญ!)
    lines.append("")
    lines.append(f"[3] Inpainting ด้วย {inpaint_name} (mask ระดับตัวอักษร)")
    uncovered = 0
    for tb in text_blocks:
        b = tb.bbox.clip(w, h).to_int()
        reg = text_mask[b[1]:b[3], b[0]:b[2]]
        if reg.size == 0:
            continue
        cover = float((reg > 0).mean())
        flag = "  ← ⚠ mask ไม่คลุมข้อความนี้!" if cover < 0.05 else ""
        if cover < 0.05:
            uncovered += 1
        lines.append(f"    OCR box {b} '{tb.source_text[:24]}' -> mask ครอบ {cover*100:.0f}%{flag}")
    lines.append(f"    รวมบล็อกที่ mask ไม่คลุมเลย: {uncovered} "
                 f"(ถ้ามาก -> สาเหตุความไม่เนียนมาจาก Detection/Segmenter)")

    # ---------------- (4) Typesetting (+ translation) ----------------
    from scanlation.typesetting import Typesetter
    ts = Typesetter(cfg.typesetting)
    if not args.no_translate and cfg.api_key():
        print("(4) Translation ...")
        try:
            from scanlation.translation import LLMTranslator
            page = LLMTranslator(cfg.translation).translate_page(page)
            lines.append("")
            lines.append(f"[4] Translation ({cfg.translation.backend}/{cfg.translation.model})")
            for bub in page.bubbles:
                lines.append(f"    {bub.uid}: '{bub.source_text}' -> '{bub.translation}'")
        except Exception as exc:  # noqa: BLE001
            lines.append("")
            lines.append(f"[4] Translation ล้มเหลว: {type(exc).__name__}: {exc}")
    else:
        # ใส่คำแปลจำลอง = ข้อความต้นฉบับ เพื่อทดสอบ typesetting อย่างเดียว
        for bub in page.bubbles:
            if not bub.translation:
                bub.set_translation(f"[TH] {bub.source_text}"[:40])
        lines.append("")
        lines.append("[4] ข้ามการแปล (ไม่มี API key หรือ --no-translate) "
                     "— ใช้ข้อความจำลองเพื่อทดสอบ typesetting")

    final = ts.render(cleaned, page)
    image_io.save_image(final, out_dir / "4_final.png")

    # ---------------- สรุป ----------------
    lines.append("")
    lines.append("=== สรุปสาเหตุที่เป็นไปได้ ===")
    if uncovered > len(text_blocks) * 0.3:
        lines.append("  ⚠ Detection/Segmenter: mask ไม่คลุมข้อความหลายจุด "
                     "-> ลอง SCANLATION_SEGMENTER=sam (ต้องมี checkpoint) หรือลด merge_gap")
    if len([b for b in page.bubbles if b.kind == BlockKind.SFX]) and uncovered:
        lines.append("  ℹ SFX (เสียงเอฟเฟกต์) มักถูกอ่านเป็น noise — พิจารณาตั้ง kind=sfx เอง")
    lines.append("  (ดูภาพ 1_ocr / 2_masks / 3_cleaned เพื่อยืนยันด้วยตา)")

    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-6:]))
    print(f"\n✓ ผลลัพธ์อยู่ใน: {out_dir}/  (0_original, 1_ocr, 2_masks, 2b_text_mask, "
          f"3_cleaned, 4_final, report.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
