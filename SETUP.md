# SETUP — คู่มือติดตั้งและตั้งค่า Environment

คู่มือนี้พาจาก **เครื่องเปล่า → รัน pipeline ได้จริง** ครบทุกขั้นตอน
ครอบคลุม Windows / macOS / Linux และทั้งแบบ **มี GPU** กับ **ใช้ CPU**

---

## 🚀 เริ่มต้นเร็ว: ใช้ผู้ให้บริการ AI "ฟรี" (ไม่ต้องมีโมเดล/บัตรเครดิต)

ยังไม่มีโมเดล AI เก่ง ๆ ใช้? เริ่มจากผู้ให้บริการที่มี **ระดับฟรี** ก่อนได้เลย
(ค่าดีฟอลต์ของโปรเจกต์ตั้งเป็น Gemini ฟรีไว้แล้ว) — 3 คำสั่ง:

```bash
cp .env.example .env                          # Windows: copy .env.example .env
#   เปิด .env ใส่คีย์ฟรีที่บรรทัด GEMINI_API_KEY=...
#   (ขอคีย์ฟรีได้ที่ https://aistudio.google.com/app/apikey)

python scripts/check_env.py                    # ตรวจว่าพร้อม
python -m scanlation.cli run 1                 # ลองแปลหน้าแรก
```

หรือให้ระบบสร้าง `.env` ให้อัตโนมัติ:

```bash
python -m scanlation.cli setup --list          # ดูผู้ให้บริการทั้งหมด (ฟรีขึ้นก่อน)
python -m scanlation.cli setup --free          # ใช้ Gemini (แนะนำ)
python -m scanlation.cli setup --provider groq # เปลี่ยนเป็น Groq
```

| ผู้ให้บริการฟรี | ขอคีย์ | จุดเด่น |
|---|---|---|
| **gemini** (ค่าดีฟอลต์) | https://aistudio.google.com/app/apikey | คุณภาพไทยดี, โควต้าฟรีรายวัน |
| **groq** | https://console.groq.com/keys | เร็วมาก เหมาะงานจำนวนมาก |
| **openrouter** | https://openrouter.ai/keys | โมเดลฟรีลงท้าย `:free` หลายตัว |
| **cerebras** | https://cloud.cerebras.ai/ | ฟรี + เร็ว |
| **mistral** | https://console.mistral.ai/api-keys/ | มีระดับฟรี |
| **together** | https://api.together.xyz/settings/api-keys | โมเดล `-Free` |
| **github** | https://github.com/settings/tokens | ใช้ GitHub PAT |
| **ollama** | — (รันในเครื่อง) | ฟรี 100% ไม่ต้องใช้คีย์ |

> 💡 เมื่อพร้อมใช้โมเดลระดับสูง ค่อยเปลี่ยนเป็น `claude-3-5-sonnet` / `gpt-4o`
> ได้โดยแก้แค่ `SCANLATION_TRANSLATION_BACKEND` + ใส่คีย์ (โค้ดไม่ต้องแก้เลย)

---

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
| **API key** (Anthropic/OpenAI/…) | ⭕ ไม่บังคับ | ขาดได้ ถ้าไม่ใช้สเตจแปล (demo/ดาวน์โหลดยังใช้ได้) — ตั้งใน `.env` |

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

💡 **ตั้งค่าเร็วสุด (3 คำสั่ง)** — คัดลอกตัวอย่าง `.env` แล้วใส่ AI key:
```bash
cp .env.example .env        # Windows: copy .env.example .env
# เปิด .env แล้วใส่ ANTHROPIC_API_KEY=...  (รายละเอียดในข้อ 4)
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

### 3.6 (ทางเลือก) SAM สำหรับ Detection — ไม่ต้องมีก็รันได้

ขั้น Detection จะเลือกตัวตัดขอบเขตบอลลูนให้อัตโนมัติ (SAM → FastSAM → Box)
ถ้ายังไม่ติดตั้ง SAM ระบบจะถอยไปใช้ `box` (กรอบสี่เหลี่ยมจาก OCR) ให้ทันที
จึง**รันได้เลยโดยไม่ต้องมี SAM** — แต่ติดตั้งแล้วจะได้ mask ที่แนบรูปบอลลูนมากขึ้น

```bash
# (ทางเลือก) ติดตั้ง SAM + ดาวน์โหลด weights
pip install -r requirements-ml.txt
python scripts/download_weights.py --model vit_b

# หรือปิดการใช้ SAM ชั่วคราวด้วยการตั้งค่าใน .env
echo "SCANLATION_SEGMENTER=box" >> .env
```

### 3.7 (ทางเลือก) LaMa สำหรับ Inpainting — ไม่ต้องมีก็รันได้

ขั้นลบข้อความเดิมจะเลือกวิธีให้อัตโนมัติ ถ้ายังไม่ติดตั้ง LaMa ระบบจะถอยให้เอง

> ⚠️ **`simple-lama-inpainting` ติดตั้งได้เฉพาะ Python 3.9–3.12** (ตรึง `pillow<10`/`numpy<2`)
> ถ้าใช้ Python 3.13+ (เช่น macOS) ให้ใช้ **`lama-torch`** ที่โหลดโมเดล TorchScript ตรง ๆ

```bash
# ✅ ทางที่แนะนำเมื่อใช้ Python 3.13+ (ได้คุณภาพ LaMa จริง)
pip install torch torchvision          # ถ้ายังไม่มี
python scripts/download_weights.py --lama   # โหลด big-lama.pt (~205 MB)
echo "SCANLATION_INPAINT_METHOD=lama-torch" >> .env
#  หรือปล่อย method=lama ไว้ก็ได้ (ระบบจะเลือก lama-torch ให้เองเมื่อไม่มี simple)

# ทางอื่น
pip install -r requirements-ml.txt     # เฉพาะ Python 3.9–3.12
pip install iopaint && echo "SCANLATION_INPAINT_METHOD=iopaint" >> .env
echo "SCANLATION_INPAINT_METHOD=median" >> .env   # เร็วสุด แต่คุณภาพต่ำ (ชั่วคราว)
```

> ℹ️ **ความเร็ว**: GPU (CUDA) / Apple Silicon (MPS) เร็วมาก (~1–3 วิ/หน้า)
> ส่วน CPU จะช้ากว่ามาก — ระบบเลือกอุปกรณ์ให้อัตโนมัติ (cuda → mps → cpu)
> ถ้าอยากเลือกเอง: `SCANLATION_INPAINT_DEVICE=mps` (macOS) หรือ `cpu`

---

## 4) ตั้งค่าด้วยไฟล์ `.env` (แนะนำ — ตั้งได้ทุกตัว)

โปรเจกต์มีไฟล์ตัวอย่าง **`.env.example`** ที่รวมค่าตั้งได้ **ทุกตัว** ไว้แล้ว
สร้างไฟล์ใช้งานจริงด้วย:

```bash
cp .env.example .env
```

จากนั้นเปิด `.env` แล้วแก้ค่าที่ต้องการ — ระบบจะ **โหลดให้อัตโนมัติ** เมื่อรัน
(ทั้ง CLI, `pipeline_example.py` และ `ScanlationConfig.load()`)

> 🔒 `.env` ถูกกันไว้ใน `.gitignore` แล้ว (ไม่ถูก commit) ส่วน `.env.example` ปลอดภัยที่จะ commit

### ลำดับความสำคัญของค่า (ตัวที่อยู่บนจะทับตัวล่าง)
```
environment variable ในเชลล์   ← สูงสุด (ใช้ชั่วคราว/CI ได้)
        ↓
ไฟล์ .env                      ← แนะนำสำหรับตั้งค่าประจำเครื่อง
        ↓
ไฟล์ config/config.json        ← ค่าดีฟอลต์ของโปรเจกต์
        ↓
ค่าดีฟอลต์ในโค้ด               ← ต่ำสุด
```

### ตัวอย่าง `.env` ขั้นต่ำ (เริ่มด้วยผู้ให้บริการฟรี)
```dotenv
# --- AI ฟรี (สเตจแปล) ---
GEMINI_API_KEY=ใส่คีย์ฟรีของคุณ
SCANLATION_TRANSLATION_BACKEND=gemini
SCANLATION_TRANSLATION_MODEL=gemini-3.8-flash

# --- อุปกรณ์ (cpu | cuda | mps) ---
SCANLATION_PROFILE=gpu
```

หรือสร้างให้อัตโนมัติ: `python -m scanlation.cli setup --free`

### สลับไปใช้โมเดลระดับสูง (เมื่อพร้อม)
```dotenv
SCANLATION_TRANSLATION_BACKEND=anthropic
SCANLATION_TRANSLATION_MODEL=claude-3-5-sonnet-latest
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

### ตัวช่วยสุดสะดวก: `SCANLATION_PROFILE`
ตั้งค่าเดียว ปรับอุปกรณ์ทั้งระบบให้เอง:
```dotenv
SCANLATION_PROFILE=cpu     # SAM+Inpaint = cpu, เลือก SAM รุ่น vit_b ให้เอง
SCANLATION_PROFILE=gpu     # SAM+Inpaint = cuda
```
หรือจะตั้งอุปกรณ์เดียวใช้ทั้งระบบด้วย `SCANLATION_DEVICE=cuda` ก็ได้
(ถ้าตั้ง `SCANLATION_SAM_DEVICE` / `SCANLATION_INPAINT_DEVICE` แยก จะทับค่านี้)

### 🔑 API key ของผู้ให้บริการ AI ทุกตัว (ใส่เฉพาะเจ้าที่ใช้)

> 🌟 **ผู้ให้บริการฟรี** อยู่บนสุด — เริ่มจากตัวใดตัวหนึ่งได้เลย
> สร้าง `.env` อัตโนมัติ: `python -m scanlation.cli setup --provider <ชื่อ>`

| ตัวแปร | ผู้ให้บริการ | ฟรี? | ตัวอย่างโมเดล |
|---|---|---|---|
| `GEMINI_API_KEY` | Google Gemini | ✅ | `gemini-3.8-flash`, `gemini-3.8-pro` |
| `GROQ_API_KEY` | Groq | ✅ | `llama-3.3-70b-versatile` |
| `OPENROUTER_API_KEY` | OpenRouter | ✅ | `meta-llama/llama-3.3-70b-instruct:free` |
| `CEREBRAS_API_KEY` | Cerebras | ✅ | `llama-3.3-70b` |
| `MISTRAL_API_KEY` | Mistral | ✅ | `mistral-small-latest` |
| `TOGETHER_API_KEY` | Together AI | ✅ | `...-Instruct-Turbo-Free` |
| `GITHUB_TOKEN` | GitHub Models | ✅ | `gpt-4o-mini` |
| (ไม่ต้องใช้คีย์) | Ollama (ในเครื่อง) | ✅ | `llama3.1` |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) | 💰 | `claude-3-5-sonnet-latest` |
| `OPENAI_API_KEY` | OpenAI (GPT) | 💰 | `gpt-4o`, `gpt-4.1` |
| `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | Azure OpenAI | 💰 | `gpt-4o` |
| `DEEPSEEK_API_KEY` | DeepSeek | 💰 | `deepseek-chat` |
| `XAI_API_KEY` | xAI (Grok) | 💰 | `grok-2-latest` |
| `PERPLEXITY_API_KEY` | Perplexity | 💰 | `sonar` |
| `COHERE_API_KEY` | Cohere | 💰 | `command-r-plus` |
| `FIREWORKS_API_KEY` | Fireworks AI | 💰 | — |
| `HF_TOKEN` | Hugging Face (โหลดโมเดล) | ✅ | — |
| `REPLICATE_API_TOKEN` | Replicate | 💰 | — |

**วิธีเลือกผู้ให้บริการ:** ตั้ง `SCANLATION_TRANSLATION_BACKEND` ให้ตรงกับเจ้าของ key
เช่น
```dotenv
SCANLATION_TRANSLATION_BACKEND=deepseek
SCANLATION_TRANSLATION_MODEL=deepseek-chat
DEEPSEEK_API_KEY=sk-xxxxxxxx
```
> ผู้ให้บริการที่ไม่ใช่ Anthropic จะเชื่อมผ่าน API แบบ **OpenAI-compatible**
> โดยระบบรู้จัก base_url ของแต่ละเจ้าให้แล้ว (DeepSeek/Groq/OpenRouter/Mistral/xAI/…)

### รายการค่าตั้งทั้งหมด
ดูได้ใน `.env.example` (จัดเป็นหมวด: API keys / Profile / Translation / Detection /
Inpainting / Typesetting / Download / Paths) — มีให้ตั้ง **ทุกตัว** รวม 60+ ค่า
หรือรันคำสั่งนี้เพื่อดูว่ามีตัวไหนตั้งอยู่แล้ว:
```bash
python -c "from scanlation.env import env_report, ENV_MAP; print(len(ENV_MAP),'ตัวที่ตั้งได้'); print(env_report())"
```

### ทางเลือก: ตั้งผ่าน environment variable โดยตรง (ไม่ใช้ .env)

**🐧🍎 macOS / Linux — ชั่วคราว:**
```bash
export GEMINI_API_KEY="คีย์ฟรีของคุณ"
export SCANLATION_PROFILE=gpu
```
**ถาวร** — ใส่บรรทัด `export ...` ใน `~/.bashrc` (หรือ `~/.zshrc`) แล้ว `source ~/.bashrc`

**🪟 Windows — ถาวร:**
```powershell
setx ANTHROPIC_API_KEY "sk-ant-xxxxxxxx"
setx SCANLATION_PROFILE "gpu"
```
**ชั่วคราว:** `$env:ANTHROPIC_API_KEY = "..."` (PowerShell) หรือ `set ANTHROPIC_API_KEY=...` (cmd)

### ✅ ตรวจว่า .env ถูกโหลดแล้ว
```bash
python scripts/check_env.py            # ดูหัวข้อ "0) ไฟล์ .env และ environment variable"
python -c "from scanlation.config import ScanlationConfig; c=ScanlationConfig.from_env(); print(c.translation.backend, c.api_key() and 'มี key' or 'ไม่มี key')"
```

> 🔐 **ความปลอดภัย:** อย่า commit `.env` ลง git — โปรเจกต์กันไว้แล้ว
> และอย่าแปะ key ลงใน issue/แชท ถ้า key หลุดให้ revoke แล้วออกใหม่ทันที

---

## 5) ตั้งค่าไฟล์ config (ไม่บังคับ — ปรับผ่าน `.env` ได้แทน)

> 💡 ปัจจุบันค่าทุกตัวของ `config.json` ตั้งผ่าน `.env` ได้หมดแล้ว (ดูข้อ 4)
> `config.json` ยังใช้ได้อยู่ และจะมีผลก็ต่อเมื่อ env/.env ไม่ได้ตั้งค่านั้นทับ

ไฟล์ `config/config.json` มีค่าดีฟอลต์พร้อมใช้ แต่ควรปรับ 2–3 จุดนี้ตามเครื่อง:

| ค่า | ปรับเมื่อ |
|---|---|
| `detection.sam_device` | `"cuda"` ถ้ามี GPU, `"cpu"` ถ้าไม่มี |
| `inpainting.device` | เหมือนด้านบน |
| `detection.sam_model_type` | `"vit_h"` (แม่นสุด) / `"vit_b"` (เบา) |
| `translation.backend` + `translation.model` | เลือก `anthropic` หรือ `openai` |
| `detection.ocr_engine` | `"paddleocr"` (แม่นสุด แนะนำ — ต้อง `pip install paddlepaddle paddleocr`) / `"easyocr"` / `"tesseract"` |

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

### 6.5 เปิดหน้า Preview/แก้ไขผลการแปล (ไม่ต้องมี GPU)
```bash
pip install -r requirements-web.txt       # Flask (เบา)
python -m scanlation.cli preview          # → http://127.0.0.1:8000
```
ควรเห็นหน้าเว็บแสดงผลการแปล เลือกหน้าได้ ลากย้าย/ปรับขนาดข้อความ และเรนเดอร์ใหม่ได้

**เครื่องมือในหน้า preview:**
- **✋ เลือก/ย้าย** — แก้คำแปล, ลากย้าย, ย่อ/ขยายกรอบ, ฟอนต์/สี/จัดชิด/ชนิดกล่อง
- **➕ เพิ่มบอลลูน** — ลากวาดกรอบใหม่บนภาพ
- **🗑 ลบบอลลูน** — ปุ่ม Delete หรือปุ่ม ✕ ที่มุมกรอบ
- **🖌 แปรงลบพื้นหลัง** — วาดทับส่วนที่อยากลบ → กด "🪄 ลบพื้นที่ที่แปรง (AI)" ให้โมเดล
  inpaint ลบให้ (มี **↶ Undo**)

> ⚠️ ถ้าเปิดหน้าที่ขึ้น **404** แปลว่าหน้านั้นยังไม่ได้ประมวลผล (ไม่มีข้อมูลใน `.scanlation/`)
> — กดปุ่ม **"▶ ประมวลผลหน้านี้"** ในหน้าเว็บ หรือรัน `python -m scanlation.cli run <n>`

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

# --- ตั้งค่า (คัดลอกตัวอย่าง .env แล้วแก้ค่า) ---
cp .env.example .env
#   แก้ในไฟล์ .env:  ANTHROPIC_API_KEY=..., SCANLATION_PROFILE=gpu  เป็นต้น

# --- ตรวจสอบ ---
python scripts/check_env.py
python pipeline_example.py --demo

# --- ใช้งาน ---
python -m scanlation.cli download "<chapter_url>" --limit 3   # [0] ดาวน์โหลด → WebP
python -m scanlation.cli run                                   # [1-4] แปล + จัดวาง
python -m scanlation.cli detect 1                              # ทดสอบ SAM + เซฟ mask
python -m scanlation.cli sources                               # ดู source ที่รองรับ
python -m scanlation.cli models                                # ดูรายชื่อโมเดลที่ใช้ได้จริง

# --- Preview/แก้ไขผลการแปล (frontend) ---
pip install -r requirements-web.txt
python -m scanlation.cli preview                               # http://127.0.0.1:8000
```

---

## 8) แก้ปัญหาที่พบบ่อย (Troubleshooting)

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `Cannot open resource` เวลาวาดฟอนต์ | path ฟอนต์ผิด | ตรวจ `typesetting.regular_font` = `fonts/Sarabun-Regular.ttf` |
| หน้า preview ขึ้น `/media/... 500 Internal Server Error` | (แก้แล้ว) Flask ตีความ relative path เทียบ `app.root_path` ไม่ใช่ cwd | อัปเดตโค้ดล่าสุด (ใช้ absolute path ใน `send_file`) |
| สระ/วรรณยุกต์ไทยวางผิดตำแหน่ง | Pillow ไม่มี libraqm | `python -c "from PIL import features; print(features.check('raqm'))"` ต้องเป็น `True` → ติดตั้ง libraqm (ข้อ 3.5) |
| `ModuleNotFoundError: segment_anything` | ยังไม่ติดตั้ง SAM | `pip install -r requirements-ml.txt` **หรือ** ตั้ง `SCANLATION_SEGMENTER=box` ใน `.env` เพื่อรันโดยไม่ต้องใช้ SAM |
| `FileNotFoundError: weights/sam_vit_h_4b8939.pth` | ติดตั้ง SAM แล้วแต่ยังไม่มีไฟล์ checkpoint | `python scripts/download_weights.py --model vit_b` (หรือ `--model vit_h`) — ระบบจะถอยไปใช้ `box` ให้เองถ้าไฟล์ยังไม่มี |
| `ModuleNotFoundError: simple_lama_inpainting` | ยังไม่ติดตั้ง LaMa | `pip install -r requirements-ml.txt` หรือใช้ `SCANLATION_INPAINT_METHOD=median` |
| `Failed to build 'pillow'` ตอนลง `simple-lama-inpainting` | แพ็กเกจตรึง `pillow<10`/`numpy<2` (ใช้ได้เฉพาะ Python 3.9–3.12) | ตั้ง `SCANLATION_INPAINT_METHOD=median` (หรือ `opencv`/`iopaint`) — หรือใช้ venv Python 3.12 |
| `ERROR: ... simple-lama-inpainting` บน Python 3.13+ | เหตุผลเดียวกับข้างบน | แนะนำ `pip install iopaint` แล้วตั้ง `SCANLATION_INPAINT_METHOD=iopaint` |
| `ต้องติดตั้ง LaMa: pip install simple-lama-inpainting` | เรียกสเตจ inpaint โดยยังไม่มี LaMa | ติดตั้งตามด้านบน หรือใช้ `SCANLATION_INPAINT_METHOD=median` ชั่วคราว |
| SAM ช้ามากตอนเริ่มหน้าแรก | SAM encode ภาพทั้งหน้า | ใช้ `vit_b` หรือเปลี่ยนเป็น `FastSAMSegmenter` |
| `CUDA out of memory` | GPU ไม่พอ | ตั้ง `sam_device = "cpu"` หรือใช้ `vit_b` |
| `401 / authentication_error` | API key ผิด/ไม่ตั้ง | ตั้งใน `.env` (ดูข้อ 4) แล้วรัน `python scripts/check_env.py` |
| `404 model not found` / `no longer available` (เช่น Gemini เปลี่ยนรุ่น) | ชื่อโมเดลเลิกให้บริการ | `python -m scanlation.cli models` เพื่อดูชื่อที่ใช้ได้ แล้วตั้ง `SCANLATION_TRANSLATION_MODEL=<ชื่อใหม่>` ใน `.env` |
| `cannot write mode RGBA as JPEG` | บันทึกภาพ RGBA เป็น JPEG | โปรเจกต์จัดการให้แล้ว (แปลงเป็น RGB) — ถ้าเจอ ให้อัปเดต Pillow |
| ดาวน์โหลดได้ไม่ครบทุกหน้า | เว็บเปลี่ยนโครงสร้าง / หน้าโดนล็อก | ดู `.scanlation/download_manifest.json` ว่า URL ไหนล้มเหลว; ปรับ `source_regex` หรือเพิ่ม cookie |
| `pip install segment-anything` ค้าง/ล้มเหลว | ต้องมี git + คอมไพล์ | ติดตั้ง git ก่อน; ถ้ามี GPU ให้ลง torch เรียบร้อยก่อน |
| `ModuleNotFoundError: flask` | ยังไม่ติดตั้ง Flask (โหมด preview) | `pip install -r requirements-web.txt` |
| เปิดหน้า preview แล้ว `/api/page/<n>` ตอบ 404 | หน้านั้นยังไม่ได้ประมวลผล | กดปุ่ม "▶ ประมวลผลหน้านี้" ในหน้าเว็บ หรือ `python -m scanlation.cli run <n>` |
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
