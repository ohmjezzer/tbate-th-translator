# ARCHITECTURE — Scanlation Engine v2

เอกสารอธิบาย "เหตุผลการออกแบบ" (design rationale) ของสถาปัตยกรรม เพื่อให้ทีม
ขยายผล/บำรุงรักษาได้ระยะยาว

---

## 1) หลักการออกแบบ (Design Principles)

| หลักการ | การนำไปใช้ในโปรเจกต์ |
|---------|----------------------|
| **Separation of Concerns** | แยกเป็น 5 ส่วน: `schemas` (ข้อมูล) / `config` (ค่า) / สเตจ 4 ตัว (ตรรกะ) / `pipeline` (วงจร) / `utils` |
| **Program to an Interface** | ทุกสเตจมี ABC: `TextDetector`, `BubbleSegmenter`, `Inpainter`, `Translator` → สลับ implementation ได้ |
| **Dependency Injection** | `ScanlationPipeline(...)` รับคอมโพเนนต์เข้าทาง constructor → mock/สลับได้ในเทสต์และ production |
| **Open/Closed + Registry** | เพิ่มโมเดลใหม่ = เขียนคลาส + `@register(...)` โดยไม่แก้ pipeline |
| **Pure data contracts** | ข้อมูลไหลเวียนผ่าน `schemas.py` เท่านั้น ทุกสเตจ serialize เป็น JSON ได้ |
| **Inspectable & Resumable** | ทุกสเตจ dumps intermediate ลง `.scanlation/*.json` → debug/แก้ไข/รันต่อได้ |
| **Lazy Initialization** | `ScanlationPipeline` สร้างคอมโพเนนต์ (SAM/LaMa/LLM) เมื่อ "เรียกใช้ครั้งแรก" เท่านั้น → คำสั่ง `download` ไม่ต้องโหลดโมเดลหนัก |
| **Fail-safe** | โมดูล ML หนัก (torch/easyocr/sam) ถูก import แบบ lazy → import แพ็กเกจได้แม้ยังไม่ติดตั้งโมเดล |

---

## 2) แผนภาพคลาส (Class Diagram)

```
                          ┌────────────────────────────┐
                          │        <<ABC>>             │
                          │       ChapterSource        │
                          └────────────┬───────────────┘
                                       │ implements
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
 VortexScansSource            GenericJsonSource              GenericHtmlSource
       │
       └─► ChapterDownloader ──► ImageConverter (บังคับ WebP) ──► orig/page-XXXX.webp

                          ┌────────────────────────────┐
                          │        <<ABC>>             │
                          │       TextDetector         │
                          └────────────┬───────────────┘
                                       │ implements
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
      EasyOCRDetector          TesseractDetector        (ต่อขยาย: PaddleOCR…)

                          ┌────────────────────────────┐
                          │        <<ABC>>             │
                          │      BubbleSegmenter       │
                          └────────────┬───────────────┘
                                       │ implements
              ┌────────────────────────┼───────────────────────┐
              ▼                        ▼                       ▼
     BoxPromptSegmenter  ◄──extends── SAMSegmenter        FastSAMSegmenter
     (fallback/no-ML)               (override _predict_mask)

                          ┌────────────────────────────┐
                          │        <<ABC>>             │
                          │        Inpainter           │
                          └────────────┬───────────────┘
                                       │ implements
        ┌──────────────────┬───────────┴────────┬──────────────────┐
        ▼                  ▼                     ▼                  ▼
 LamaInpainter   IOPaintLaMaInpainter   OpenCVInpainter   MedianFillInpainter

                          ┌────────────────────────────┐
                          │        <<ABC>>             │
                          │        Translator          │
                          └────────────┬───────────────┘
                                       │ implements
                                  LLMTranslator
                          uses ──► LLMBackend { AnthropicBackend | OpenAIBackend }

                          ┌────────────────────────────┐
                          │        Typesetter          │
                          └────────────────────────────┘
                          uses ──► ThaiTextWrapper + FontFitter + PillowRenderer

  ── Orchestrator ────────────────────────────────────────────────────────────
                       ┌───────────────────────────┐
                       │    ScanlationPipeline     │  (DI: detector, segmenter,
                       └───────────────────────────┘   inpainter, translator, typesetter)
```

---

## 3) โครงสร้างข้อมูล (Data Flow)

```
Page
 ├── index, image_path, width, height
 └── bubbles: Bubble[]
       ├── uid, bbox (BBox), kind (BlockKind), polygon, mask (np.ndarray)
       ├── blocks: TextBlock[]     ← บรรทัดข้อความต้นฉบับ (จาก OCR)
       └── translation            ← คำแปลไทย (จากสเตจ 3) เขียนลง blocks[0].translation

BBox      : x0,y0,x1,y1 + width/height/area/center, expand(), clip(), iou(), union()
TextBlock : uid, bbox, source_text, confidence, kind, translation, meta
ImageRef  : index, url, page_label, meta          ← สเตจ 0 (ก่อนดาวน์โหลด)
```

- `kind` (BlockKind): `speech | narration | sfx | credit | unknown`
- `mask` ถูกเก็บใน JSON เป็น **RLE** (`encode_rle`/`decode_rle`) เพื่อประหยัดขนาด

---

## 4) ลำดับการทำงาน (Sequence) — `ScanlationPipeline`

```
caller ──download(chapter_url)          [สเตจ 0 — ไม่โหลดโมเดลหนัก]
  │
  └─(0) ChapterSource.list_images(url) ─► ImageRef[]
           └─► ThreadPool: GET(url) ─► ImageConverter.to_webp ─► orig/page-XXXX.webp
           └─ manifest: .scanlation/download_manifest.json

caller ──process_page(image_path)
  │
  ├─(1) DetectionPipeline.detect_page(image)
  │        OCR → TextBlock[]  ─►  BubbleSegmenter.segment() → Bubble[] (mask+kind)
  │        └─ dump: .scanlation/1_detection_pXXXX.json
  │
  ├─(3) Translator.translate_page(page)          # แปลก่อน inpaint (ต้องอ่านข้อความเดิม)
  │        build_system_prompt(context) + build_user_prompt(bubbles)
  │        ──► LLMBackend.complete() ──► {id, th}[]  ──► Bubble.set_translation()
  │        └─ dump: .scanlation/3_translation_pXXXX.json   (+ cache)
  │
  ├─(2) build_text_mask(image, bubbles) + Inpainter.inpaint()
  │        └─ save: .scanlation/2_cleaned_pXXXX.png
  │
  └─(4) Typesetter.render(cleaned, page)  ──►  save: out/page-XXXX.png
```

> เหตุผลที่สลับลำดับเป็น 1 → 3 → 2 → 4: การแปลต้องใช้ "ข้อความต้นฉบับ" ที่ยังอยู่บนภาพ
> และ `mask` ที่ได้จากสเตจ 1 ถูกใช้ต่อในทั้งสเตจ 2 และ 4

---

## 5) คู่มือขยายผล (Extension Guide)

### 5.0 เพิ่ม Download Source ใหม่ (เช่น เว็บอื่น)

```python
# scanlation/downloader/mysite.py
import re
from ..utils.registry import register
from .base import ChapterSource, ImageRef

@register("source", "mysite")
class MySiteSource(ChapterSource):
    def __init__(self, config):
        self.cfg = config
    def list_images(self, chapter_url):
        html = self.fetch_text(chapter_url)
        urls = re.findall(r'https://cdn\.mysite\.com/[^"\']+\.(?:webp|jpg|png)', html)
        return [ImageRef(index=i, url=u, page_label=f"page-{i:04d}")
                for i, u in enumerate(urls, 1)]
```

แล้วตั้ง `cfg.download.source = "mysite"` — ตัว `ChapterDownloader` จะบังคับ
แปลงเป็น WebP และตั้งชื่อไฟล์ให้เองโดยอัตโนมัติ (ไม่ต้องแก้ downloader)

### 5.1 เพิ่มโมเดล Inpainting ใหม่ (เช่น MAT, MI-GAN)

```python
# scanlation/inpainting/my_inpainter.py
from ..utils.registry import register
from .base import Inpainter

@register("inpainter", "mat")
class MATInpainter(Inpainter):
    def __init__(self, config): ...
    def inpaint(self, image, mask):
        ...
        return cleaned
```

จากนั้นตั้งใน config — **ไม่ต้องแก้ pipeline เลย**:
```python
cfg.inpainting.method = "mat"
```
(หมายเหตุ: ต้องผูก `_default_inpainter()` ให้รู้จักชื่อ หรือใช้ registry `build()` —
ดูจุดต่อใน `pipeline.py`)

### 5.2 เพิ่ม Segmenter ใหม่ (SAM2 / MobileSAM)

สืบทอด `BoxPromptSegmenter` แล้ว **override เฉพาะ `_predict_mask()`**:

```python
@register("segmenter", "sam2")
class SAM2Segmenter(BoxPromptSegmenter):
    def _predict_mask(self, image, box):
        # เรียก SAM2 predictor แทน
        ...
```

### 5.3 เพิ่มผู้ให้บริการ LLM ใหม่ (เช่น Gemini)

```python
class GeminiBackend(LLMBackend):
    def complete(self, system, user) -> dict:
        ...
```
แล้ว `LLMTranslator(cfg, backend=GeminiBackend(cfg))`

### 5.4 เพิ่มฟอนต์/สไตล์

แก้ `typesetting` ใน config หรือ override `Typesetter._style_for()` เพื่อทำ
สไตล์ตามตัวละคร (เช่น ตัวโกงใช้ฟอนต์อีกแบบ)

---

## 6) เหตุผลการเลือกเทคโนโลยี

| ทางเลือก | ที่เลือกใช้ | เหตุผล |
|----------|-------------|--------|
| Download | **requests + ThreadPoolExecutor** | เบา, มี retry/stream, ดาวน์โหลดขนานได้ |
| Image format | **WebP (บังคับ)** | ลดขนาดไฟล์, ทำให้ downstream ทำงานสม่ำเสมอ, คงคุณภาพ |
| Segmentation | **SAM (vit_h/vit_l/vit_b)** | mask แนบรูปทรงบอลลูนจริง, prompt ด้วย box จาก OCR ได้ตรง ๆ |
| (ทางเลือกเบา) | **FastSAM** | รันบน CPU ได้, เร็วกว่า, เหมาะ deploy ไม่มี GPU |
| Inpainting | **LaMa** | ขึ้นชื่อเรื่อง large-mask, ให้พื้นหลังเนียนต่อเนื่อง |
| (ทางเลือกอื่น) | **IOPaint** | รวมหลายโมเดล (MAT/MI-GAN) → ขยายได้โดยไม่เขียน adapter ใหม่ |
| Translation | **Claude 3.5 Sonnet / GPT-4o** | คุณภาพภาษาอดีเยี่ยม + รองรับ structured JSON output |
| Typesetting | **Pillow + pythainlp** | ควบคุม layout ได้ละเอียด + ตัดคำไทยได้ถูกต้อง |
| Registry | **dict + decorator** | เบา, ไม่พึ่งไลบรารี DI ภายนอก |

---

## 7) การทดสอบ (Testing Strategy)

- **Unit test ที่ไม่ต้องมี ML** — ใช้ `BoxPromptSegmenter`, `MedianFillInpainter`,
  `DummyTranslator` (ดู `pipeline_example.py::run_demo`) → รันได้บน CI ทุกเครื่อง
- **Contract test** — ทุก implementation ของ interface เดียวกันต้องรับ/คืนชนิดเดิม
- **Golden test** — เก็บ `.scanlation/*.json` ของหน้าตัวอย่างไว้เทียบผล

```python
# ตัวอย่าง contract test
def test_inpainter_contract():
    from scanlation.inpainting import MedianFillInpainter
    out = MedianFillInpainter().inpaint(np.zeros((10,10,3),np.uint8), np.ones((10,10),bool))
    assert out.shape == (10,10,3)
```

---

## 8) แผนขยายในอนาคต (Roadmap)

1. **Download ซับซ้อนขึ้น** — login/session, cloudflare bypass, ดึงทั้งเรื่อง (ทุก chapter) พร้อมคิว
2. **SAM2 / video tracking** — ลดภาระการ re-detect เมื่อบอลลูนอยู่ตำแหน่งเดิม
2. **Comic-text-detector** (YOLO) แทน OCR เพื่อหาข้อความที่ OCR อ่านไม่ออก
3. **Batch API + async** — แปลหลายหน้า/หลายเรื่องพร้อมกัน
4. **Web UI / queue** — ตรวจและแก้ mask/คำแปลแบบคลิกได้
5. **Glossary auto-learn** — สกัดชื่อเฉพาะจากเรื่องทั้งเรื่องอัตโนมัติ
6. **Fine-tune ฟอนต์มังงะไทย** — สำหรับหัวข้อที่ต้อง "สวยแบบการ์ตูน"
