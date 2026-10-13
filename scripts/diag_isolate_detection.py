#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diag_isolate_detection.py
=================================
การทดลองแบบ "ควบคุมตัวแปร" (A/B) เพื่อพิสูจน์ว่าความไม่เนียนมาจากสเตจไหน

ใช้ภาพเดียวกัน + inpainter ตัวเดียวกัน + typesetter ตัวเดียวกัน
เปลี่ยน "เฉพาะผลลัพธ์ของสเตจ Detection" เท่านั้น:

    A) detection จาก pipeline จริง (ตามที่โค้ดทำงานอยู่ = merge ข้ามหน้า)
    B) detection "ที่ถูกต้อง" (กรอบแยกต่อบรรทัด/ต่อพื้นที่ ที่มนุษย์กำหนดเอง)

ถ้าผลลัพธ์ B เนียน แต่ A ไม่เนียน  => ต้นตออยู่ที่ Detection ไม่ใช่ LaMa/Pillow

    python scripts/diag_isolate_detection.py <image> [--inpaint median|opencv|lama-torch]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanlation.config import ScanlationConfig  # noqa: E402
from scanlation.env import apply_env, load_dotenv  # noqa: E402
from scanlation.inpainting import build_text_mask  # noqa: E402
from scanlation.schemas import BBox, BlockKind, Bubble, TextBlock  # noqa: E402
from scanlation.typesetting import Typesetter  # noqa: E402
from scanlation.utils import image_io  # noqa: E402

# --- กรอบ "ความจริง" ที่มนุษย์กำหนด (golden) สำหรับหน้าทดสอบนี้ ---
GROUPS = [
    # (kind, [(x0,y0,x1,y1), ...] ต่อบรรทัด, ข้อความ)
    (BlockKind.SPEECH, [(148, 146, 366, 187), (148, 196, 392, 237)],
     "I WILL PROTECT EVERYONE I LOVE!"),
    (BlockKind.NARRATION, [(108, 506, 421, 547), (108, 556, 387, 597)],
     "But the twin horns were nowhere to be found."),
]


def _bubbles(groups, w: int, h: int):
    out = []
    for i, (kind, boxes, text) in enumerate(groups):
        bb = [BBox(*b) for b in boxes]
        uni = bb[0]
        for b in bb[1:]:
            uni = uni.union(b)
        blocks = [TextBlock(uid=f"t{i}-{j}", bbox=b, source_text="", confidence=1.0)
                  for j, b in enumerate(bb)]
        bub = Bubble(uid=f"gold{i}", bbox=uni, kind=kind, blocks=blocks)
        bub.set_translation(text)          # ข้อความไทย/ยาว Approximation เพื่อทดสอบ typeset
        out.append(bub)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="diag/isolate")
    ap.add_argument("--inpaint", default="median")
    args = ap.parse_args()

    load_dotenv()
    cfg = apply_env(ScanlationConfig())
    cfg.inpainting.method = args.inpaint

    image = image_io.load_image_rgb(Path(args.image))
    h, w = image.shape[:2]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from scanlation.pipeline import ScanlationPipeline
    pipe = ScanlationPipeline(cfg)
    inpainter = pipe.inpainter
    ts = Typesetter(cfg.typesetting)

    # ---------- A) detection จริงจาก pipeline ----------
    page_a = pipe.detection.detect_page(image, index=0)
    for b in page_a.bubbles:
        if not b.translation:
            b.set_translation(f"(A) {b.source_text}"[:60])
    mask_a = build_text_mask(image, page_a.bubbles, cfg.inpainting)
    clean_a = inpainter.inpaint(image, mask_a)
    final_a = ts.render(clean_a, page_a)
    image_io.save_image(clean_a, out / "A_clean.png")
    image_io.save_image(final_a, out / "A_final.png")

    # ---------- B) detection ที่ถูกต้อง (golden) ----------
    page_b = type(page_a)(index=0, image_path=args.image, width=w, height=h)
    page_b.bubbles = _bubbles(GROUPS, w, h)
    mask_b = build_text_mask(image, page_b.bubbles, cfg.inpainting)
    clean_b = inpainter.inpaint(image, mask_b)
    final_b = ts.render(clean_b, page_b)
    image_io.save_image(clean_b, out / "B_clean.png")
    image_io.save_image(final_b, out / "B_final.png")

    # ---------- ตัวเลขเปรียบเทียบ ----------
    def stats(page, mask):
        rows = []
        for b in page.bubbles:
            box = b.bbox
            cov = (mask[int(box.y0):int(box.y1), int(box.x0):int(box.x1)] > 0)
            rows.append((b.uid, b.kind.value, box.to_int(),
                         int(box.area) or 1, int(cov.sum()),
                         f"{cov.mean()*100:.0f}%", b.translation[:26]))
        return rows

    rep = [f"ISOLATE-DETECTION  image={args.image} ({w}x{h}) inpainter={args.inpaint}", ""]
    rep.append(f"mask พิกเซลรวม: A={int((mask_a>0).sum())}  B={int((mask_b>0).sum())}")
    for tag, page, mask in (("A (pipeline)", page_a, mask_a), ("B (golden)", page_b, mask_b)):
        rep.append(f"\n--- {tag} ---")
        for uid, kind, box, area, cov, pct, tr in stats(page, mask):
            rep.append(f"  {uid:8s} [{kind:9s}] box={box} พื้นที่={area} "
                       f"maskในกรอบ={cov}px ({pct}) '{tr}'")
    rep.append("\nสรุป: ถ้า B_clean สะอาดกว่า A_clean อย่างชัดเจน "
               "=> ต้นตอคือ Detection (กรอบรวมกันผิด) ไม่ใช่ LaMa/Pillow")
    (out / "isolate_report.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"\n✓ {out}/A_clean.png A_final.png B_clean.png B_final.png isolate_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
