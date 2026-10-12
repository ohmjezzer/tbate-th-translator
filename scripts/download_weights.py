# -*- coding: utf-8 -*-
"""
scripts/download_weights.py
===========================
ดาวน์โหลด checkpoint ของโมเดลที่ต้องใช้ (SAM) ไปไว้ที่โฟลเดอร์ weights/

    python scripts/download_weights.py            # ดาวน์โหลด SAM ViT-H (ค่าเริ่มต้น)
    python scripts/download_weights.py --model vit_b   # เวอร์ชันเล็ก (เบา/เร็ว)

หมายเหตุ: LaMa ผ่าน `simple-lama-inpainting` จะมีน้ำหนักโมเดลติดมากับแพ็กเกจอยู่แล้ว
          จึงไม่ต้องดาวน์โหลดเพิ่ม
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"มีไฟล์อยู่แล้ว: {dest}")
        return
    print(f"กำลังดาวน์โหลด {url}\n  → {dest}")

    def progress(block_num, block_size, total_size):
        done = block_num * block_size
        if total_size > 0:
            pct = min(100, done * 100 // total_size)
            print(f"\r  {pct:3d}%  ({done/1e6:.0f}/{total_size/1e6:.0f} MB)", end="")
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print("\nเสร็จสิ้น")


def main() -> None:
    ap = argparse.ArgumentParser(description="ดาวน์โหลด checkpoint ของโมเดล")
    ap.add_argument("--model", default="vit_h", choices=list(SAM_URLS),
                    help="ขนาดของ SAM (vit_h/vit_l/vit_b)")
    ap.add_argument("--dir", default="weights", help="โฟลเดอร์ปลายทาง")
    args = ap.parse_args()

    url = SAM_URLS[args.model]
    dest = Path(args.dir) / url.split("/")[-1]
    download(url, dest)


if __name__ == "__main__":
    main()
