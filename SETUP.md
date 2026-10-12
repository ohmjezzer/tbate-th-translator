# SETUP — คู่มือติดตั้งและตั้งค่า Environment

คู่มือนี้พาจาก **เครื่องเปล่า → รัน pipeline ได้จริง** ครบทุกขั้นตอน
ครอบคลุม Windows / macOS / Linux และทั้งแบบ **มี GPU** กับ **ใช้ CPU**

> 💡 ก่อนเริ่ม: คุณจะรันสคริปต์ตรวจสอบได้ทุกเมื่อด้วย
> ```bash
> python scripts/check_env.py
> ```
> มันจะบอกว่าอะไรพร้อมแล้ว (✓) อะไรยังขาด (✗) พร้อมวิธีแก้

---

## 0) สรุปสิ่งที่ต้องมี (Prerequisites)

| สิ่งที่ต้องมี | จำเป็นไหม | หมายเหตุ |
|---|---|---|
| **Python 3.11** (3.9+ ใช้ได้) | ✅ บังคับ | แนะนำ 3.11 เพราะเสถียรกับ torch/easyocr |
| **Git** | ✅ บังคับ | สำหรับ clone และ `pip install` จาก git |
| **libraqm** | ✅ บังคับ (สำหรับไทย) | ให้ Pillow จัดวางสระ/วรรณยุกต์ไทยถูกต้อง |
| **GPU NVIDIA + CUDA** | ⭕ ไม่บังคับ | ไม่มีก็รันได้ แต่ SAM/LaMa จะช้า (ใช้ CPU) |
| **Tesseract** | ⭕ ไม่บังคับ | ต้องมีเฉพาะถ้าตั้ง `ocr_engine: "tesseract"` |
| **API key** (Anthropic/OpenAI) | ⭕ ไม่บังคับ | ขาดได้ ถ้าไม่ใช้สเตจแปล (demo/ดาวน์โหลดยังใช้ได้) |

> ⚠️ **เรื่องเวอร์ชัน Python:** โปรเจกต์นี้ทดสอบบน Python 3.11 เป็นหลัก
> ถ้าใช้ 3.13/3.14 อาจมีบางแพ็กเกจ (easyocr, torch บางรุ่น) ยังไม่รองรับ
> แนะนำสร้าง venv ด้วย 3.11 ถ้ามีให้เลือก

---

## 1) ขั้นแรก: โคลนโปรเจกต์

```bash
git clone https://github.com/ohmjezzer/tbate-th-translator.git
cd tbate-th-translator
git checkout v2-upgrade        # สาขาที่มี Scanlation Engine v2
```

> ถ้ายังไม่ merge เข้า `main` ให้ใช้สาขา `v2-upgrade` ตามด้านบน

---

## 2) สร้างสภาพแวดล้อม (เลือกทางใดทางหนึ่ง)

### 🅰️ ทางเลือก A — venv (เบา, แนะนำถ้าใช้ Python ที่ลงไว้แล้ว)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# ถ้าโดน policy บล็อก: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows (cmd):**
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

เมื่อ activate สำเร็จ จะเห็น `(.venv)` นำหน้า prompt
ปิดได้ด้วยคำสั่ง `deactivate`

### 🅱️ ทางเลือก B — Conda (แนะนำถ้าต้องการจัดการ CUDA/Python ให้ครบ)

```bash
conda env create -f environment.yml
conda activate scanlation
```

`environment.yml` จะติดตั้ง Python 3.11 + libraqm + tesseract + แพ็กเกจต่าง ๆ ให้
(ต้องมี Anaconda/Miniconda ก่อน — ดาวน์โหลดจาก https://docs.conda.io/en/latest/miniconda.html)

---

## 3) ติดตั้ง Dependency

> ทำตามลำดับ — ของพื้นฐานก่อน แล้วค่อยของหนัก (โมเดล ML)

### 3.1 ของพื้นฐาน (บังคับ)

```bash
pip install -r requirements.txt
```
ได้แก่: numpy, Pillow, opencv-python-headless, pythainlp, requests, PyYAML,
easyocr, pytesseract, anthropic, openai

### 3.2 PyTorch (จำเป็นสำหรับ SAM/LaMa)

เลือกคำสั่งให้ตรงกับเครื่อง:

**มี GPU NVIDIA (ตรวจ CUDA ของเครื่องก่อนด้วย `nvidia-smi`):**
```bash
# ตัวอย่าง CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
**ใช้ CPU อย่างเดียว:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
> ดูคำสั่งที่ตรงกับเครื่องคุณได้ที่ https://pytorch.org/get-started/locally/

### 3.3 โมเดล ML (SAM, LaMa, FastSAM)

```bash
pip install -r requirements-ml.txt
```
ได้แก่: segment-anything (Meta SAM), simple-lama-inpainting (LaMa),
ultralytics (FastSAM), iopaint

### 3.4 ดาวน์โหลด Checkpoint ของ SAM

```bash
python scripts/download_weights.py --model vit_h    # ใหญ่สุด แม่นสุด (~2.4 GB)
python scripts/download_weights.py --model vit_b    # เบา/เร็ว (~375 MB) แนะนำเริ่มตรงนี้
```
ไฟล์จะไปอยู่ที่ `weights/sam_*.pth`
> **LaMa ไม่ต้องดาวน์โหลดเพิ่ม** — น้ำหนักโมเดลติดมากับแพ็กเกจแล้ว

### 3.5 ติดตั้ง libraqm และ Tesseract (ระดับระบบ)

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libraqm-dev tesseract-ocr
```
**macOS (Homebrew):**
```bash
brew install libraqm tesseract
```
**Windows:** Pillow เวอร์ชันใหม่มี libraqm ให้ในตัว — ถ้า `check_env.py` รายงานว่าไม่มี
ให้ลอง `pip install -U pillow` และใช้ Conda (`conda install -c conda-forge pillow`)

---

## 4) ตั้งค่า API key (สำหรับสเตจแปล)

โปรเจกต์อ่าน key จาก **environment variable** (ไม่เก็บในโค้ด/ไฟล์ config เพื่อความปลอดภัย)

| ตัวแปร | ใช้เมื่อ |
|---|---|
| `ANTHROPIC_API_KEY` | `translation.backend = "anthropic"` (Claude 3.5 Sonnet) |
| `OPENAI_API_KEY` | `translation.backend = "openai"` (GPT-4o) |

### 🪟 Windows

**ตั้งแบบถาวร (ต้องเปิด terminal ใหม่):**
```powershell
setx ANTHROPIC_API_KEY "sk-ant-xxxxxxxx"
```
**ตั้งชั่วคราว (ใช้เฉพาะ session นี้):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-xxxxxxxx"     # PowerShell
set ANTHROPIC_API_KEY=sk-ant-xxxxxxxx          # cmd
```

### 🐧🍎 macOS / Linux

**ตั้งชั่วคราว (ใช้เฉพาะ session):**
```bash
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxx"
```
**ตั้งแบบถาวร** — ใส่บรรทัดนี้ในไฟล์ `~/.bashrc` (หรือ `~/.zshrc`):
```bash
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxx"
```
แล้วรัน `source ~/.bashrc` (หรือเปิด terminal ใหม่)

### ✅ ตรวจว่า key ถูกตั้งแล้ว
```bash
python -c "import os; print('มี key' if os.environ.get('ANTHROPIC_API_KEY') else 'ยังไม่มี')"
```

> 🔐 **ความปลอดภัย:** อย่า commit key ลง git — โปรเจกต์มี `.gitignore` ที่กัน `.env` ไว้แล้ว
> ถ้าใช้หลาย key/สลับ provider บ่อย แนะนำตั้งเป็น script `env.sh` (ที่ gitignore ไว้)

---

## 5) ตั้งค่าไฟล์ config (ไม่บังคับ)

ไฟล์ `config/config.json` มีค่าดีฟอลต์พร้อมใช้ แต่ควรปรับ 2–3 จุดนี้ตามเครื่อง:

| ค่า | ปรับเมื่อ |
|---|---|
| `detection.sam_device` | `"cuda"` ถ้ามี GPU, `"cpu"` ถ้าไม่มี |
| `inpainting.device` | เหมือนด้านบน |
| `detection.sam_model_type` | `"vit_h"` (แม่นสุด) / `"vit_b"` (เบา) |
| `translation.backend` + `translation.model` | เลือก `anthropic` หรือ `openai` |
| `detection.ocr_engine` | `"easyocr"` (แนะนำ) หรือ `"tesseract"` |

สร้างไฟล์ config ตัวอย่างใหม่ได้ด้วย:
```bash
python -m scanlation.cli init-config --dir config
```
ตรวจ system prompt ที่จะส่งให้ LLM (เช็คว่าบริบทตัวละครถูกต้อง):
```bash
python -m scanlation.cli show-prompt
```

---

## 6) ตรวจสอบว่าติดตั้งสำเร็จ

### 6.1 รันสคริปต์ตรวจสภาพแวดล้อม
```bash
python scripts/check_env.py
```
ควรเห็น `✓` ครบในส่วนที่บังคับ และสรุปท้ายว่า "สภาพแวดล้อมพื้นฐานพร้อมใช้งาน 🎉"

### 6.2 ทดสอบแบบไม่ต้องใช้ API key / ไม่ต้องมี GPU
```bash
python pipeline_example.py --demo
```
ถ้าสำเร็จจะได้ไฟล์ `out/demo_before_after.png` (ภาพเทียบก่อน/หลัง)

### 6.3 ทดสอบดาวน์โหลด (ไม่ต้องใช้ API key)
```bash
python -m scanlation.cli download "https://vortexscans.org/series/the-beginning-after-the-end/chapter-255" --limit 2
```
ควรได้ `orig/page-0001.webp`, `orig/page-0002.webp`

### 6.4 ทดสอบเต็มระบบ (ต้องมี API key + SAM weights)
```bash
python -m scanlation.cli run 1 2      # แปลเฉพาะหน้า 1, 2
```
ควรได้ `out/page-0001.png`, `out/page-0002.png`

---

## 7) คำสั่งสรุปแบบเร็ว (Cheat Sheet)

```bash
# --- สร้าง env (venv) ---
python3.11 -m venv .venv && source .venv/bin/activate      # macOS/Linux
.\.venv\Scripts\Activate.ps1                               # Windows

# --- ติดตั้ง ---
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-ml.txt
python scripts/download_weights.py --model vit_h
sudo apt-get install -y libraqm-dev tesseract-ocr          # ระบบ (Linux)

# --- ตั้งค่า ---
export ANTHROPIC_API_KEY="sk-ant-xxxx"                     # macOS/Linux
setx ANTHROPIC_API_KEY "sk-ant-xxxx"                       # Windows (ถาวร)

# --- ตรวจสอบ ---
python scripts/check_env.py
python pipeline_example.py --demo

# --- ใช้งาน ---
python -m scanlation.cli download "<chapter_url>" --limit 3   # [0] ดาวน์โหลด → WebP
python -m scanlation.cli run                                   # [1-4] แปล + จัดวาง
python -m scanlation.cli detect 1                              # ทดสอบ SAM + เซฟ mask
python -m scanlation.cli sources                               # ดู source ที่รองรับ
```

---

## 8) แก้ปัญหาที่พบบ่อย (Troubleshooting)

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `Cannot open resource` เวลาวาดฟอนต์ | path ฟอนต์ผิด | ตรวจ `typesetting.regular_font` = `fonts/Sarabun-Regular.ttf` |
| สระ/วรรณยุกต์ไทยวางผิดตำแหน่ง | Pillow ไม่มี libraqm | `python -c "from PIL import features; print(features.check('raqm'))"` ต้องเป็น `True` → ติดตั้ง libraqm (ข้อ 3.5) |
| `ModuleNotFoundError: segment_anything` | ยังไม่ติดตั้ง SAM | `pip install -r requirements-ml.txt` |
| `ModuleNotFoundError: simple_lama_inpainting` | ยังไม่ติดตั้ง LaMa | `pip install -r requirements-ml.txt` |
| `ต้องติดตั้ง LaMa: pip install simple-lama-inpainting` | เรียกสเตจ inpaint โดยยังไม่มี LaMa | ติดตั้งตามด้านบน หรือใช้ `cfg.inpainting.method = "median"` ชั่วคราว |
| SAM ช้ามากตอนเริ่มหน้าแรก | SAM encode ภาพทั้งหน้า | ใช้ `vit_b` หรือเปลี่ยนเป็น `FastSAMSegmenter` |
| `CUDA out of memory` | GPU ไม่พอ | ตั้ง `sam_device = "cpu"` หรือใช้ `vit_b` |
| `401 / authentication_error` | API key ผิด/ไม่ตั้ง | ตั้ง API key ใหม่ (ข้อ 4) |
| `cannot write mode RGBA as JPEG` | บันทึกภาพ RGBA เป็น JPEG | โปรเจกต์จัดการให้แล้ว (แปลงเป็น RGB) — ถ้าเจอ ให้อัปเดต Pillow |
| ดาวน์โหลดได้ไม่ครบทุกหน้า | เว็บเปลี่ยนโครงสร้าง / หน้าโดนล็อก | ดู `.scanlation/download_manifest.json` ว่า URL ไหนล้มเหลว; ปรับ `source_regex` หรือเพิ่ม cookie |
| `pip install segment-anything` ค้าง/ล้มเหลว | ต้องมี git + คอมไพล์ | ติดตั้ง git ก่อน; ถ้ามี GPU ให้ลง torch เรียบร้อยก่อน |
| Tesseract ไม่พบ | ไม่ได้ติดตั้งโปรแกรม | `sudo apt install tesseract-ocr` (หรือ `brew install tesseract`) |

---

## 9) หมายเหตุด้านทรัพยากร

| โหมด | RAM | VRAM (ถ้าใช้ GPU) | ความเร็ว |
|---|---|---|---|
| Demo (`--demo`) | ~500 MB | — | ทันที |
| ดาวน์โหลดอย่างเดียว | ต่ำมาก | — | ~1-2 วินาที/หน้า |
| Detection (SAM vit_b, CPU) | ~2–3 GB | — | ~5-10 วินาที/หน้า |
| Detection (SAM vit_h, GPU) | ~4 GB | ~6–8 GB | ~1-2 วินาที/หน้า |
| Inpainting (LaMa, GPU) | ~2 GB | ~2–3 GB | ~1 วินาที/หน้า |

> รวมทั้งหน้า (ดึง → ตรวจ → ลบ → แปล → วาง) บน GPU ระดับกลาง ~8–15 วินาที/หน้า
> ค่าใช้จ่าย LLM หลัก ๆ คือสเตจแปล (เปิด cache ไว้แล้ว จึงแปลซ้ำไม่คิดเงิน)
