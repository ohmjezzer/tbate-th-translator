# Scanlation Engine v2 — Manga Thai Translator (Professional Grade)

ระบบ **สแกนเลชัน/แปลมังงะอัตโนมัติ** เวอร์ชันอัปเกรดจากโปรเจกต์
[`tbate-th-translator`](https://github.com/ohmjezzer/tbate-th-translator) เดิม
(EasyOCR/Tesseract + OpenCV) ให้เป็น **สถาปัตยกรรมแบบโมดูลที่ขยายผลได้** พร้อม
**5 ขั้นตอน**:

| ขั้น | งาน | เทคโนโลยีใหม่ | เทียบกับของเดิม |
|-----|------|----------------|------------------|
| **0. Download & Ingest** | ดึงรูปจากเว็บ → บังคับ WebP → `orig/` | **ChapterSource + ImageConverter** (requests + Pillow) | เดิมดาวน์โหลดมือ |
| **1. Detection & Masking** | หาข้อความ + ตัดขอบเขตบอลลูน | **SAM (Segment Anything Model, Meta)** / FastSAM | เดิมใช้กรอบสี่เหลี่ยมแข็ง ๆ |
| **2. Inpainting** | ลบข้อความต้นฉบับ | **LaMa** (Large-Mask Inpainting) / IOPaint | เดิมใช้ `cv2.inpaint` |
| **3. Translation** | แปลเป็นไทย | **Claude 3.5 Sonnet / GPT-4o** + Character Context | เดิมกรอกคำแปลมือใน `data.py` |
| **4. Typesetting** | จัดวางข้อความไทย | **Pillow** + Sarabun + Thai word wrap | เดิมมีแล้ว แต่ปรับให้ clip ในบอลลูน |

> เป้าหมาย: ผลลัพธ์ที่ "เนียนตา" เหมือนทีมสแกนเลชันมืออาชีพทำเอง ไม่ใช่แค่ OCR แล้ววางทับ

---

## 1) ทำไมต้องอัปเกรด? (สิ่งที่เปลี่ยนไป)

**ของเดิม (สคริปต์แบนราบ)**
- `extract2.py` อ่านข้อความด้วย Tesseract → ได้กรอบสี่เหลี่ยม
- `data.py` เก็บพิกัด + คำแปล (แก้มือทีละหน้า)
- `render.py` ใช้ `cv2.inpaint` บนกรอบสี่เหลี่ยม แล้ววาดข้อความทับ
- ปัญหา: mask เป็นสี่เหลี่ยม → ลบเละ, แปลไม่สม่ำเสมอ, ขยายยาก

**เวอร์ชันใหม่ (แพ็กเกจ `scanlation/`)**
- **SAM** ให้ mask ที่แนบตามรูปทรงบอลลูนจริง (วงรี ขอบหยัก หางบอลลูน) → ลบสะอาด
- **LaMa** เติมพื้นหลังให้เนียนต่อเนื่อง (ไม่เหลือรอย)
- **LLM + Character Context** แปลทั้งหน้าโดยรู้ว่าใครพูดกับใคร → สรรพนาม/น้ำเสียงคงเส้นคงวา
- **Typesetting** ที่ clip ข้อความให้อยู่ใน mask ของบอลลูนอัตโนมัติ
- **สถาปัตยกรรมแบบ interface + dependency injection** → สลับโมเดลได้จาก config โดยไม่แตะ pipeline

---

## 2) สถาปัตยกรรม (ภาพรวม)

```
                       ScanlationPipeline (scanlation/pipeline.py)
                       ───────────────────────────────────────────
  เว็บต้นทาง ─► [0] Download & Ingest ─► [1] Detection & Masking ─► [2] Inpainting ─► [4] Typesetting ─► ภาพผลลัพธ์
  (chapter URL)  source+WebP→orig/       OCR + SAM → Bubble[]      LaMa (mask→clean)   Pillow (วาดไทย)
                                              │                                            ▲
                                              └──────────────► [3] Translation ───────────┘
                                                              Claude/GPT-4o
                                                              → Bubble.translation
```

**ลำดับการทำงานจริงใน `process_page()`** (หมายเหตุ: แปลก่อน inpaint เพราะต้องอ่านข้อความต้นฉบับ)

```
image ─► (1) detect_page → Page(bubbles + masks + kind)
      ─► (3) translate_page → เติมคำแปลในแต่ละ bubble
      ─► (2) build_text_mask + inpaint → ภาพที่ลบข้อความเดิม
      ─► (4) typesetter.render → ภาพสุดท้าย
      ─► บันทึก out/page-XXXX.png + intermediate .scanlation/*.json
```

### โครงสร้างไฟล์

```
scanlation/
├── schemas.py            # Data model กลาง: BBox / TextBlock / Bubble / Page
├── config.py             # ค่า config ทั้งหมด (dataclass) + โหลด JSON/YAML + API key
├── utils/
│   ├── image_io.py       # อ่าน/เขียนภาพ (มาตรฐานเดียวทั้งโปรเจกต์)
│   ├── geometry.py       # iou / merge / mask / polygon
│   ├── registry.py       # ★ plugin registry → สลับโมเดลด้วยชื่อใน config
│   └── logging_utils.py
├── downloader/           # [0] ดึงรูปจากเว็บ → บังคับ WebP
│   ├── base.py           #   interface: ChapterSource, ImageRef
│   ├── vortexscans.py    #   VortexScansSource (เว็บตัวอย่าง)
│   ├── generic.py        #   GenericJsonSource, GenericHtmlSource (เว็บอื่น)
│   ├── converter.py      #   ImageConverter (บังคับ WebP)
│   └── downloader.py     #   ChapterDownloader (orchestrator + manifest)
├── detection/            # [1] OCR + SAM
│   ├── base.py           #   interface: TextDetector, BubbleSegmenter
│   ├── ocr_detector.py   #   EasyOCRDetector, TesseractDetector
│   ├── sam_segmenter.py  #   BoxPromptSegmenter, SAMSegmenter, FastSAMSegmenter
│   └── pipeline.py       #   DetectionPipeline
├── inpainting/           # [2] ลบข้อความเดิม
│   ├── base.py           #   interface: Inpainter
│   ├── masks.py          #   สร้าง mask ระดับตัวอักษร/บอลลูน
│   ├── lama_inpainter.py #   LamaInpainter, IOPaintLaMaInpainter
│   └── opencv_inpainter.py # OpenCVInpainter, MedianFillInpainter (fallback)
├── translation/          # [3] แปลด้วย LLM
│   ├── base.py           #   interface: Translator
│   ├── glossary.py       #   CharacterProfile / Glossary / StoryContext
│   ├── prompts.py        #   ★ System Prompt + User Prompt + JSON schema
│   └── llm_translator.py #   LLMTranslator + AnthropicBackend + OpenAIBackend
├── typesetting/          # [4] จัดวางข้อความไทย
│   ├── base.py           #   TextStyle / TextLayout
│   ├── thai_wrapper.py   #   ThaiTextWrapper (ตัดบรรทัด ไม่ตัดกลางคำ)
│   ├── fitter.py         #   FontFitter (binary-search ขนาดฟอนต์)
│   ├── renderer.py       #   PillowRenderer (auto-contrast + clip ใน mask)
│   └── typesetter.py     #   Typesetter (orchestrator ของสเตจ)
├── pipeline.py           # ★ ScanlationPipeline — เชื่อมทุกสเตจ
└── cli.py                # command-line interface

pipeline_example.py       # ตัวอย่างเชื่อมทั้ง 5 ขั้นตอน (รันได้จริง)
scripts/download_weights.py   # ดาวน์โหลด checkpoint ของ SAM
scripts/check_env.py          # ตรวจว่าติดตั้ง environment ครบไหม
config/                   # characters.json / glossary.json / config.json
fonts/                    # Sarabun-Regular.ttf / Sarabun-Bold.ttf
.env.example              # ตัวอย่างค่าตั้งทุกตัว (คัดลอกเป็น .env)
README.md / SETUP.md / ARCHITECTURE.md
requirements.txt / requirements-ml.txt / environment.yml
```

---

## 3) การติดตั้ง (Conda / Virtual Environment)

> 📖 **คู่มือติดตั้งแบบละเอียดทีละขั้น (Windows/macOS/Linux + GPU/CPU) อยู่ใน [`SETUP.md`](SETUP.md)**
> และตรวจว่าติดตั้งครบไหมได้ด้วย `python scripts/check_env.py`

### ⚙️ ตั้งค่าทุกอย่างผ่าน `.env` — เริ่มจาก AI "ฟรี" ได้ทันที

โปรเจกต์ตั้งค่าดีฟอลต์เป็น **ผู้ให้บริการ AI ที่มีระดับฟรี** (Gemini) และมี
**`.env.example`** ที่รวมค่าตั้งได้ทุกตัว (paths / download / detection /
inpainting / translation / typesetting) **รวม API key ของผู้ให้บริการ AI ทุกเจ้า**

```bash
# วิธีที่ 1: คัดลอกแล้วใส่คีย์ (เริ่มจาก Gemini ฟรี)
cp .env.example .env      # แล้วใส่ GEMINI_API_KEY=... (ขอฟรี: https://aistudio.google.com/app/apikey)

# วิธีที่ 2: ให้ระบบสร้าง .env ให้
python -m scanlation.cli setup --free            # ใช้ Gemini (แนะนำ)
python -m scanlation.cli setup --provider groq   # หรือเลือกเจ้า อื่น
python -m scanlation.cli setup --list            # ดูผู้ให้บริการทั้งหมด
```

**ผู้ให้บริการฟรีที่รองรับ** (ไม่ต้องใช้บัตรเครดิต): `gemini`, `groq`, `openrouter`,
`cerebras`, `mistral`, `together`, `github`, `ollama` (รันในเครื่อง)
**ระดับเสียเงิน** (คุณภาพสูงสุด): `anthropic`, `openai`, `deepseek`, `xai`, `perplexity`, `cohere`, `azure`, `fireworks`

สลับผู้ให้บริการทำได้แค่แก้ 2 บรรทัดใน `.env` (โค้ดไม่ต้องแตะ):
```dotenv
SCANLATION_TRANSLATION_BACKEND=groq
SCANLATION_TRANSLATION_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...
```
ค่าทุกตัวโหลดจาก `.env` อัตโนมัติ โดยมีลำดับความสำคัญ:
`env ในเชลล์ > ไฟล์ .env > config/config.json > ค่าดีฟอลต์`

### ทางเลือก A: venv (แนะนำสำหรับเครื่องเดียว)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 1) ของพื้นฐาน
pip install -r requirements.txt

# 2) ของหนัก (โมเดล ML) — ติดตั้ง torch ให้ตรงกับ CUDA ก่อน
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-ml.txt

# 3) ดาวน์โหลด checkpoint ของ SAM
python scripts/download_weights.py --model vit_h       # หรือ vit_b (เบากว่า)

# 4) (ถ้าใช้ Tesseract) ติดตั้งระดับระบบ
sudo apt-get install -y tesseract-ocr libraqm-dev
```

### ทางเลือก B: Conda (แนะนำสำหรับจัดการ CUDA/Python 3.11)

```bash
conda env create -f environment.yml
conda activate scanlation
python scripts/download_weights.py
```

### ⚠️ สิ่งสำคัญที่มักลืม

- **libraqm**: Pillow ต้องคอมไพล์ด้วย **libraqm** เพื่อจัดวางสระ/วรรณยุกต์ไทย
  ให้ถูกตำแหน่ง (`pip install pillow` รุ่นใหม่มักมีให้มาแล้ว ถ้าไม่มี:
  `pip install pillow --upgrade` หรือใช้ conda `pillow` จาก conda-forge)
  ทดสอบ: `python -c "from PIL import features; print(features.check('raqm'))"` ต้องได้ `True`
- **GPU/CPU**: ตั้ง `detection.sam_device` และ `inpainting.device` ให้ตรงเครื่อง
  (ใช้ `cpu` ได้แต่ช้ามาก)
- **API key**: `export ANTHROPIC_API_KEY=...` หรือ `OPENAI_API_KEY=...`

---

## 4) วิธีใช้

### 4.1 ลองแบบเดโม (ไม่ต้องมี GPU/API key)

```bash
python pipeline_example.py --demo
# → ได้ out/demo_before_after.png (ภาพเทียบก่อน/หลัง)
```

### 4.2 รันเต็มระบบ

```bash
# วางภาพต้นฉบับไว้ที่ orig/page-0001.webp, page-0002.webp, ...
export ANTHROPIC_API_KEY=sk-ant-...

python -m scanlation.cli run            # แปลทุกหน้า
python -m scanlation.cli run 4 23 26    # เฉพาะบางหน้า
python -m scanlation.cli detect 4       # ทดสอบเฉพาะ Detection + เซฟภาพ mask
python -m scanlation.cli show-prompt    # ตรวจ system prompt ที่จะส่งให้ LLM
python -m scanlation.cli init-config    # สร้าง config ตัวอย่าง
python scripts/check_env.py             # ตรวจว่าติดตั้งครบไหม
```

ผลลัพธ์: `out/page-XXXX.png` และไฟล์ intermediate ใน `.scanlation/`

### 4.3 ดาวน์โหลดรูปจากเว็บ (สเตจ 0)

```bash
# ดาวน์โหลด chapter → บังคับแปลงเป็น WebP → วางใน orig/ (พร้อมให้ pipeline อ่านต่อ)
python -m scanlation.cli download "https://vortexscans.org/series/the-beginning-after-the-end/chapter-255"

# ตัวเลือกที่มีให้ใช้
python -m scanlation.cli download <url> --limit 3            # ทดลอง 3 หน้าแรก
python -m scanlation.cli download <url> --out orig --workers 6 --quality 92
python -m scanlation.cli download <url> --source vortexscans # ระบุ source
python -m scanlation.cli download <url> --run                # ดาวน์โหลดเสร็จแล้วแปลต่อทันที
python -m scanlation.cli sources                             # ดู source ที่รองรับ
```

**รับประกันว่าไฟล์ที่ได้เป็น WebP เสมอ** — ต่อให้ต้นทางเป็น `.jpg/.png/.webp`
ก็จะถูกแปลงเป็น `orig/page-0001.webp, page-0002.webp, …` เรียงตามลำดับหน้า
และเขียน `manifest` ไว้ที่ `.scanlation/download_manifest.json` (เก็บ URL ต้นทาง
ของทุกหน้า ไว้ตรวจสอบ/ทำซ้ำได้)

ผ่านโค้ด Python:
```python
from scanlation.config import ScanlationConfig
from scanlation.pipeline import ScanlationPipeline

cfg = ScanlationConfig()
cfg.download.source = "vortexscans"
cfg.download.max_workers = 6
cfg.download.limit = 0

pipe = ScanlationPipeline(cfg)
report = pipe.download("https://vortexscans.org/series/<slug>/chapter-255")
print(report.summary())          # [download] vortexscans: 41/41 หน้า ...

# หรือดาวน์โหลด + แปลต่อในคำสั่งเดียว
pipe.download_and_run("https://vortexscans.org/series/<slug>/chapter-255")
```

### 4.4 ใช้ผ่านโค้ด (Python API)

```python
from scanlation.config import ScanlationConfig
from scanlation.pipeline import ScanlationPipeline

cfg = ScanlationConfig()
cfg.detection.sam_device = "cuda"
cfg.inpainting.method = "lama"
cfg.translation.backend = "anthropic"

pipeline = ScanlationPipeline(cfg)          # สร้างคอมโพเนนต์จาก config อัตโนมัติ
results = pipeline.run(pages=[4, 23, 26], do_translate=True)
for r in results:
    print(r.page_index, "->", r.output_path)
```

---

## 5) เจาะลึกแต่ละขั้นตอน (พร้อมตำแหน่งโค้ด)

### [0] Download & Ingest — `scanlation/downloader/`

- `base.py` → interface `ChapterSource` (บอกว่ารูปมีอะไรบ้าง) + `ImageRef`
- `vortexscans.py::VortexScansSource` → ดึงหน้า chapter แล้วสกัด URL รูปด้วย regex
  (จาก HTML จริง: รูปอยู่บน `storage.vortexscans.org/.../page-0001_...webp`)
- `generic.py` → `GenericJsonSource` (อ่าน URL จากไฟล์ JSON/API) และ
  `GenericHtmlSource` (จับ URL รูปด้วย regex ปรับได้) — **เพิ่มเว็บใหม่ได้โดยไม่เขียนโค้ด**
- `converter.py::ImageConverter` → **บังคับแปลงเป็น WebP** ด้วย Pillow (จัดการ RGBA/P,
  flatten พื้นขาว, ย่อขนาดได้) ควบคุม `quality`/`method`
- `downloader.py::ChapterDownloader` → ดาวน์โหลดแบบ **หลายเธรด**, retry,
  ตั้งชื่อ `page-XXXX.webp`, เขียน manifest

**วิธีเพิ่มเว็บใหม่** (ตัวอย่าง): ตั้งใน config
```python
cfg.download.source = "generic-html"        # หรือ "generic-json"
cfg.download.source_regex = r"https://cdn\.[^\"']+\.(webp|jpg|png)"
```
หรือเขียนคลาสใหม่ + `@register("source", "mysite")` แล้วตั้ง `cfg.download.source = "mysite"`

### [1] Detection & Masking — `scanlation/detection/`

- `ocr_detector.py` → อ่านข้อความ (`EasyOCRDetector` / `TesseractDetector`)
- `sam_segmenter.py` → จัดกลุ่มบรรทัดให้เป็นบอลลูน แล้วใช้ **SAM** ตัด mask
  - `SAMSegmenter` — ใช้ `segment_anything` (Meta) เป็นค่าเริ่มต้น
    prompt = กรอบข้อความ, เลือก mask ที่ดีสุด, ขยายผลด้วย SAM2/MobileSAM ได้
  - `FastSAMSegmenter` — ทางเลือกที่เบากว่า (ultralytics, รันบน CPU ได้)
  - `BoxPromptSegmenter` — fallback ไม่ใช้ ML (ใช้ทดสอบ/unit test)
  - จำแนกประเภทกล่อง: `speech` / `narration` / `sfx` ด้วย heuristic (ความสว่าง + ทรง)
- **ทำไม SAM**: ให้ mask ที่แนบตามรูปทรงบอลลูนจริง → ขั้น inpaint ลบได้เฉพาะตัวอักษร
  โดยไม่ทำลายเส้นขอบบอลลูน

### [2] Inpainting — `scanlation/inpainting/`

- `masks.py::build_text_mask()` — สร้าง mask ระดับ **ตัวอักษร** (เทียบสีกับพื้นหลัง
  มัธยฐาน → เก็บขอบ anti-alias ไม่เหลือ ghosting) และจำกัดให้อยู่ในบอลลูน (จาก SAM)
- `lama_inpainter.py::LamaInpainter` — **LaMa** เติมพื้นหลังให้เนียน
  (pad ขนาดให้หารด้วย 8 ลงตัวให้อัตโนมัติ)
  - `IOPaintLaMaInpainter` — เชื่อม IOPaint เพื่อใช้โมเดลอื่น (MAT / MI-GAN / SD) ในอนาคต
- `opencv_inpainter.py` — `OpenCVInpainter` (fallback เดิม) และ `MedianFillInpainter`
  (เร็วสุด เหมาะบอลลูนสีพื้นเรียบ)

### [3] Translation — `scanlation/translation/`

- `glossary.py` — นิยาม **CharacterProfile** (สรรพนาม/น้ำเสียง/วิธีเรียกคนอื่น),
  **Glossary** (คำเฉพาะ) และ **StoryContext** (เรื่องย่อ + แนวทางภาษา)
- `prompts.py` — สร้าง **System Prompt** ที่ "เน้นบริบทตัวละคร" + กติกาภาษาไทย
  และบังคับ output เป็น JSON ที่ map ตาม id ของบล็อก
- `llm_translator.py` — `LLMTranslator` (แปลทั้งหน้าในครั้งเดียว → คำสอดคล้องกัน)
  - `AnthropicBackend` (Claude 3.5 Sonnet — ใช้ tool use บังคับ JSON)
  - `OpenAIBackend` (GPT-4o — ใช้ Structured Outputs)
  - มี **cache** (ไม่แปลซ้ำ) + **retry** + **glossary enforcement**
- ตัวอย่าง config: `config/characters.json`, `config/glossary.json`

### [4] Typesetting — `scanlation/typesetting/`

- `thai_wrapper.py` — ตัดบรรทัดไทยด้วย `pythainlp.word_tokenize` (ไม่ตัดกลางคำ)
  + จัดการคำละติน (เว้นวรรคถูกต้อง) + ซอยคำที่ยาวเกินกรอบ
- `fitter.py` — **binary search** หาขนาดฟอนต์ที่ใหญ่สุดที่ยังพอดีกล่อง (2 รอบ strict→relaxed)
- `renderer.py` — วาดด้วย Pillow: จัดกึ่งกลาง, **auto-contrast** (พื้นมืด→ตัวอักษรขาว),
  **stroke** สำหรับ SFX และ **composite_clipped** ตัดข้อความไม่ให้ล้นบอลลูน
- `typesetter.py` — รวมทุกอย่าง: ระยะขอบในบอลลูน (padding) + เลือกสี + วาด

---

## 6) การย้ายจากโปรเจกต์เดิม (Migration)

| ไฟล์เดิม | แทนด้วย |
|----------|---------|
| (ดาวน์โหลดรูปด้วยมือ / สคริปต์ ad-hoc) | `scanlation/downloader/` — `python -m scanlation.cli download <url>` (บังคับ WebP อัตโนมัติ) |
| `extract.py`, `extract2.py`, `tblocks.py`, `tscan.py`, `ocr_test.py` | `scanlation/detection/` (OCR + SAM) — ลบสคริปต์ 5 ตัวรวมเป็น 1 สเตจ |
| `data.py` (พิกัด + คำแปลมือ) | `scanlation/schemas.py` (โครงสร้างอัตโนมัติ) + `config/glossary.json` + `config/characters.json` |
| `render.py` (inpaint 4 เหลี่ยม + วาดไทย) | `scanlation/inpainting/` + `scanlation/typesetting/` |
| `leftover.py`, `chk25.py` (QA) | ตรวจจาก `.scanlation/*_translation.json` ได้ (ผลลัพธ์ทุกสเตจเป็น JSON) |

> จุดที่ยังต้องมี "คนตรวจ": พิกัดบอลลูนที่ SAM พลาด และคำแปลที่เป็นชื่อเฉพาะ
> — แนะนำให้ตรวจผ่านไฟล์ JSON ใน `.scanlation/` แล้วแก้ก่อนเรนเดอร์รอบสุดท้าย

---

## 7) Tips การปรับคุณภาพ

- **บอลลูนได้ mask ผิด** → เพิ่ม `detection.sam_expand`, ลด `merge_gap` (แยกบอลลูนที่ติดกัน),
  หรือใช้ `use_auto_mask = true` (AutomaticMaskGenerator ช้ากว่าแต่ครบกว่า)
- **ยังเหลือรอยข้อความ (ghosting)** → เพิ่ม `inpainting.mask_dilate` (เช่น 8→12)
  หรือลด `inpainting.text_delta` (45→35) ให้จับตัวอักษรจางมากขึ้น
- **ข้อความไทยล้นบอลลูน** → เพิ่ม `typesetting.padding_ratio` (0.14→0.18)
  และลด `max_font_size`
- **คำแปลสรรพนามมั่ว** → เพิ่มรายละเอียดใน `config/characters.json`
  (pronouns / address_others / speech_style ให้ครบ)
- **ค่าใช้จ่าย LLM** → เปิด cache ไว้ (ดีฟอลต์) และทดสอบด้วย `--no-translate` ก่อน

---

## 8) แก้ปัญหาที่พบบ่อย

| อาการ | สาเหตุ/วิธีแก้ |
|-------|----------------|
| `cannot open resource` เวลาวาดฟอนต์ | path ฟอนต์ผิด — ตรวจ `typesetting.regular_font` |
| สระ/วรรณยุกต์ไทยวางผิดที่ | Pillow ไม่มี libraqm → `features.check('raqm')` ต้องเป็น True |
| `ModuleNotFoundError: segment_anything` | ยังไม่ติดตั้ง → `pip install -r requirements-ml.txt` |
| SAM ช้ามากตอน `set_image` | ใช้ `vit_b` หรือเปลี่ยนเป็น `FastSAMSegmenter` |
| `CUDA out of memory` | ลด `sam_model_type` เป็น `vit_b` หรือตั้ง device = `cpu` |

---

## 9) สัญญาอนุญาต / ข้อจำกัด

โค้ดนี้จัดทำเพื่อ **การศึกษา/ทดลอง** เท่านั้น ตัวการ์ตูนและภาพต้นฉบับเป็นลิขสิทธิ์
ของเจ้าของผลงาน — โปรดขออนุญาตและให้เครดิตตามที่ควร
