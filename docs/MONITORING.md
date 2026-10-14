# 📊 ระบบ Monitoring — เห็นภาพรวมปัญหาทุกสเตจของ Pipeline

> เวอร์ชัน: เพิ่มใน PR นี้ | ไฟล์หลัก: `scanlation/monitoring/`

## ปัญหาที่ระบบนี้แก้

ก่อนหน้านี้ เวลาผลลัพธ์ออกมามีบั๊ก เช่น **"แปลไม่หมด"**, **"ไทยปนอังกฤษ"**, **"ภาพไม่เนียน (ลบพื้นหลังผิด)"**
ผู้ใช้ต้องค่อย ๆ ไล่เปิด log และไฟล์ intermediate เองทีละไฟล์เพื่อหาว่า *"ปัญหาเกิดที่สเตจไหน"*

ระบบ Monitoring ทำให้ทุกครั้งที่ pipeline ประมวลผล 1 หน้า ระบบจะ **เก็บตัวชี้วัด (metrics)
ของทั้ง 4 สเตจ + สร้างภาพวินิจฉัย + แจ้งประเด็นที่น่าสงสัย (issues) อัตโนมัติ** แล้วแสดงผล
บนแดชบอร์ดในหน้าเว็บ Preview ทันที

---

## สถาปัตยกรรม

```
process_page()
   │
   ├─ (1) Detection   ──┐
   ├─ (3) Translation   │  MetricsCollector รวบรวมตัวชี้วัดทีละสเตจ
   ├─ (2) Inpainting    │  + วิเคราะห์ issues อัตโนมัติ
   ├─ (4) Typesetting ──┘
   │
   └─ (5) finalize_page()
           ├─ .scanlation/monitoring/page-XXXX.json     (metrics + issues)
           ├─ .scanlation/monitoring/page-XXXX-mask.png (มาสก์ขาวดำที่ใช้ลบ)
           └─ .scanlation/monitoring/page-XXXX-overlay.png (ภาพจริง + มาสก์แดง + กรอบบอลลูน)
```

หน้าเว็บ Preview (`python -m scanlation.cli preview`) มีแถบ **"📊 สุขภาพหน้านี้"** เพิ่มขึ้นมา
ระหว่างรายการหน้ากับภาพหลัก แสดง:

- การ์ดตัวชี้วัดของแต่ละสเตจ (Detection / Translation / Inpainting / Typesetting)
- รายการ **issues** พร้อมรหัสมาตรฐาน (แดง = ต้องแก้, เหลือง = เตือน)
- ภาพวินิจฉัย 2 ใบ (มาสก์ / กรอบตรวจจับ) คลิกดูภาพเต็มได้

---

## ตัวชี้วัดและเกณฑ์เตือน (Issues)

| รหัส Issue | ระดับ | ความหมาย | สเตจต้นตอ |
|---|---|---|---|
| `NO_TEXT_DETECTED` | 🔴 | OCR ไม่พบข้อความเลย — ภาพอาจไม่มีตัวอักษร/เบลอ | Detection |
| `LOW_CONFIDENCE` | 🟡 | มีบล็อกที่ OCR มั่นใจ < 50% — อาจจับผิด/จับไม่ครบ | Detection |
| `OVERSIZED_BUBBLE` | 🔴 | มีบอลลูนกินพื้นที่ > 60% ของภาพ — อาจรวมกรอบผิด (ต้นตอ "ภาพไม่เนียน") | Detection |
| `MISSING_TRANSLATION` | 🔴 | มีบอลลูนที่ยังไม่มีคำแปล | Translation |
| `LEFTOVER_ENGLISH` | 🔴 | คำแปลมีภาษาอังกฤษหลงเหลือ (ไทยปนอังกฤษ) | Translation |
| `EMPTY_MASK` | 🔴 | Mask ว่าง — ไม่มีอะไรถูกลบเลย | Inpainting |
| `HIGH_MASK_COVERAGE` | 🔴 | Mask ครอบคลุม > 35% ของภาพ — อาจลบพื้นหลังเกินจริง | Inpainting |
| `UNRENDERED_BUBBLES` | 🟡 | มีบอลลูนที่ไม่ได้วาดคำแปลทับ (เพราะไม่มีคำแปล) | Typesetting |

เกณฑ์ปรับได้ที่ `MetricsCollector.LOW_CONF_THRESHOLD / MAX_BUBBLE_AREA_RATIO / MAX_MASK_COVERAGE`

---

## API ที่เพิ่ม (ฝั่ง Preview server)

| Endpoint | คำอธิบาย |
|---|---|
| `GET /api/metrics` | สรุปภาพรวมทุกหน้า (จำนวนหน้า / หน้าที่มี issue / ความถี่ของรหัส issue) |
| `GET /api/metrics/<index>` | metrics เต็มของหน้าที่ระบุ (404 ถ้ายังไม่มี) |
| `GET /monitoring/<index>/<filename>` | ดึงภาพวินิจฉัย (mask/overlay) — มีการกัน path traversal |

---

## การทดสอบ

```bash
python scripts/diag_monitoring.py
```

สคริปต์นี้จำลอง 2 สถานการณ์โดยไม่ต้องใช้โมเดล/API จริง:

1. **หน้าปกติ** — คาดว่า 0 issues, มีไฟล์ JSON + ภาพ mask ครบ
2. **หน้าปัญหาครบ** — บอลลูนบวม + OCR มั่นใจต่ำ + ขาดคำแปล + ไทยปนอังกฤษ + mask ใหญ่ผิดปกติ
   คาดว่าเจอ issue ครบทั้ง 6 รหัส

และทดสอบ Flask API (`/api/metrics`, `/api/metrics/<i>`, `/monitoring/...`, กัน path traversal)
รวม **26 checks** ผ่านทั้งหมด

นอกจากนี้ยังทดสอบแบบ end-to-end (mock detection/translator/inpainter เข้ากับ
`ScanlationPipeline.process_page` จริง) ยืนยันว่าไฟล์ monitoring ถูกเขียนครบและ metrics ถูกต้อง

---

## หมายเหตุ

- ไฟล์ภาพหน้าที่ประมวลผลด้วยเวอร์ชันเก่า (ก่อนมี monitoring) จะยังไม่มี metrics —
  แดชบอร์ดจะแสดง "ยังไม่มีข้อมูล metrics" ให้กด **"ประมวลผลหน้านี้"** ใหม่เพื่อสร้าง
- ตัวตรวจ `LEFTOVER_ENGLISH` ใช้กฎเบื้องต้น (ตัวอักษรละตินยาว ≥ 3) ไม่ได้หักลบคำที่
  glossary อนุญาต — หากเจอ false positive บ่อย สามารถยกระดับให้รับ glossary ได้ในอนาคต
