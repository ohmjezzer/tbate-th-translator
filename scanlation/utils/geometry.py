# -*- coding: utf-8 -*-
"""
geometry.py
===========
ฟังก์ชันเรขาคณิตและ mask ที่ใช้บ่อยในสเตจ Detection / Inpainting
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from ..schemas import BBox


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over Union ของสองกรอบ [x0,y0,x1,y1]"""
    return BBox.from_list(a).iou(BBox.from_list(b))


def _x_overlap_ratio(a: "BBox", b: "BBox") -> float:
    """สัดส่วนการทับซ้อน "แนวนอน" เทียบกับกรอบที่แคบกว่า (0..1)"""
    iw = min(a.x1, b.x1) - max(a.x0, b.x0)
    if iw <= 0:
        return 0.0
    return iw / max(1e-9, min(a.width, b.width))


def merge_overlapping(
    boxes: List[Tuple[BBox, object]],
    gap: int = 0,
    threshold: float = 0.0,
    min_x_overlap: float = 0.0,
    max_area_ratio: float = 0.0,
) -> List[Tuple[BBox, List[object]]]:
    """
    จัดกลุ่มกรอบที่ทับซ้อน/ใกล้กันเข้าด้วยกัน (greedy union-find แบบง่าย)

    ใช้ในสเตจ Detection เพื่อรวม "บรรทัดข้อความหลายบรรทัดในบอลลูนเดียว"
    ให้กลายเป็นบอลลูนก้อนเดียว

    พารามิเตอร์กัน "การรวมผิด" (สาเหตุหลักของภาพไม่เนียน):
        min_x_overlap  : ต้องทับซ้อนแนวนอนอย่างน้อยสัดส่วนนี้ของกรอบที่แคบกว่า
                         จึงจะยอมให้รวม (0 = ปิดการตรวจ = พฤติกรรมเดิม)
                         จำเป็นมาก เพราะข้อความบอลลูน กับ SFX ที่อยู่ข้าง ๆ
                         อยู่ "แนวเดียวกัน" แต่ไม่ควรถูกรวมเป็นก้อนเดียว
        max_area_ratio : ห้ามรวมกรอบที่ขนาดต่างกันเกินอัตราส่วนนี้
                         (เช่น กรอบรวมแถบกว้างทั้งหน้ากับคำสั้น ๆ) 0 = ปิด

    คืนค่า: list ของ (BBox รวม, list ของ payload ที่อยู่ในกลุ่ม)
    """
    n = len(boxes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    def compatible(a: "BBox", b: "BBox") -> bool:
        if min_x_overlap > 0 and _x_overlap_ratio(a, b) < min_x_overlap:
            return False
        if max_area_ratio > 0:
            big, small = max(a.area, b.area), min(a.area, b.area)
            if small > 0 and big / small > max_area_ratio:
                return False
        return True

    # เตรียมกรอบที่ขยายตาม gap เพื่อวัด "ความใกล้"
    expanded = [b.expand(gap) for b, _ in boxes]
    for i in range(n):
        for j in range(i + 1, n):
            if not _near(expanded[i], expanded[j]):
                continue
            if not compatible(expanded[i], expanded[j]):
                continue
            if expanded[i].iou(expanded[j]) > threshold or _near(expanded[i], expanded[j]):
                union(i, j)

    groups: dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    result: List[Tuple[BBox, List[object]]] = []
    for members in groups.values():
        merged = boxes[members[0]][0]
        for m in members[1:]:
            merged = merged.union(boxes[m][0])
        result.append((merged, [boxes[m][1] for m in members]))
    return result


def _near(a: BBox, b: BBox) -> bool:
    """ตรวจว่าสองกรอบทับซ้อนกันหรือไม่ (หลังขยาย gap แล้ว)"""
    return not (a.x1 < b.x0 or b.x1 < a.x0 or a.y1 < b.y0 or b.y1 < a.y0)


def mask_from_boxes(
    boxes: Iterable[Sequence[float]],
    height: int,
    width: int,
    expand: int = 0,
) -> np.ndarray:
    """สร้าง binary mask (bool) จากรายการกรอบ"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        b = BBox.from_list(box).expand(expand).clip(width, height)
        x0, y0, x1, y1 = b.to_int()
        mask[max(0, y0):y1, max(0, x0):x1] = 1
    return mask.astype(bool)


def dilate_mask(mask: np.ndarray, px: int = 4) -> np.ndarray:
    """ขยาย binary mask ด้วย morphological dilation (ใช้ก่อน inpaint)"""
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    out = cv2.dilate(mask.astype(np.uint8), k, iterations=1)
    return out.astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """คืน connected component ที่ใหญ่ที่สุดของ binary mask"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx)


def polygon_from_mask(mask: np.ndarray, epsilon_ratio: float = 0.01) -> List[Tuple[int, int]]:
    """สกัด contour หลักของ mask แล้วย่อจุดให้เหลือ polygon อย่างง่าย"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    eps = epsilon_ratio * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    return [(int(p[0][0]), int(p[0][1])) for p in approx]
