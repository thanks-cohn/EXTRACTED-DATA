# BulkOCR

**BulkOCR turns screenshots and image collections into structured, auditable JSON metadata.**

It combines layout detection with OCR so it can find the relevant metadata panel inside a much larger image, extract named fields, preserve source coordinates, and process either one image or an entire directory.

BulkOCR currently extracts gallery-oriented fields such as:

- title and creator;
- date;
- group, type, and language;
- series and characters;
- tags and tag-symbol evidence;
- thumbnail bounds and dimensions;
- warnings, confidence values, and OCR diagnostics.

The source image is never modified. Output JSON is written beside the source image unless another workflow moves it afterward.

> **Current limitation:** small gender symbols attached to tag chips are preserved as OCR evidence, but exact `♀` and `♂` normalization still needs refinement.

## Install

BulkOCR runs locally on Linux, macOS, and Windows. No cloud OCR account or API key is required.

### Requirements

- Git
- Python 3.12 or 3.13 recommended
- Tesseract OCR for the standard backend
- PaddleOCR is optional and provides the higher-accuracy backend

Python 3.14 can run the standard Tesseract setup, but PaddlePaddle currently requires Python 3.13 or earlier.

## Linux

### 1. Install system packages

#### Arch Linux / Garuda

```bash
sudo pacman -S --needed git python tesseract tesseract-data-eng
```

#### Debian / Ubuntu

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip tesseract-ocr tesseract-ocr-eng
```

#### Fedora

```bash
sudo dnf install -y git python3 python3-pip tesseract tesseract-langpack-eng
```

### 2. Clone and install BulkOCR

```bash
git clone https://github.com/thanks-cohn/BulkOCR.git
cd BulkOCR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Optional: install PaddleOCR

Use Python 3.12 or 3.13:

```bash
python3.13 -m venv .venv-paddle
source .venv-paddle/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-paddle.txt
```

## macOS

Install Homebrew first if it is not already installed, then run:

```bash
brew install git python@3.13 tesseract

git clone https://github.com/thanks-cohn/BulkOCR.git
cd BulkOCR
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional PaddleOCR backend:

```bash
python -m pip install -r requirements-paddle.txt
```

## Windows

### 1. Install prerequisites

Install:

- Git for Windows
- Python 3.12 or 3.13 from python.org
- Tesseract OCR for Windows

During Python installation, enable **Add Python to PATH**.

### 2. Clone and install

Open PowerShell:

```powershell
git clone https://github.com/thanks-cohn/BulkOCR.git
Set-Location BulkOCR
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional PaddleOCR backend:

```powershell
python -m pip install -r requirements-paddle.txt
```

If PowerShell blocks virtual-environment activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Verify the installation

Standard Tesseract backend:

```bash
tesseract --version
python -c "import cv2, pytesseract, PIL; print('BulkOCR dependencies OK')"
```

PaddleOCR backend:

```bash
python -c "import paddle, paddleocr; print('PaddleOCR backend OK')"
```

## Quick start

Process one image with automatic OCR selection:

```bash
python batch_extract.py /path/to/image.png --ocr-engine auto
```

Process one image with PaddleOCR:

```bash
python batch_extract.py /path/to/image.png --ocr-engine paddle
```

Process an entire directory recursively:

```bash
python batch_extract.py /path/to/images \
  --recursive \
  --overwrite \
  --ocr-engine auto \
  --no-extract-thumbnail \
  --non-interactive
```

On Windows PowerShell:

```powershell
python batch_extract.py "C:\path\to\images" `
  --recursive `
  --overwrite `
  --ocr-engine auto `
  --no-extract-thumbnail `
  --non-interactive
```

## Output

For an input image named:

```text
example.png
```

BulkOCR writes:

```text
example-EXTRACTED-DATA.json
```

The output contains the original image path, extraction status, OCR diagnostics, detected metadata region, structured fields, tag and character items, thumbnail information, warnings, and errors.

## How it works

BulkOCR does not simply OCR the entire screenshot and guess which text matters. It uses a bounded extraction pipeline:

1. Locate and validate the metadata panel.
2. Detect the stable label stack and section geometry.
3. Stop extraction at the metadata panel boundary.
4. OCR and normalize only the relevant regions.
5. Preserve raw OCR, normalized values, coordinates, confidence, and diagnostics.
6. Write one machine-readable JSON record beside each source image.

The foundational label stack is:

```text
Group
Type
Language
Series
Characters
Tags
```

`Group` and `Tags` are the strongest anchors. The other labels strengthen confidence and define the field ranges.

## Interactive thumbnail extraction

When a thumbnail is detected, interactive mode asks:

```text
Extract thumbnail? [y/N]:
```

- `y` extracts it.
- `n` or Enter leaves it unextracted.

The default thumbnail filename is:

```text
<original-image-name-without-extension>-THUMBNAIL.png
```

The JSON records thumbnail coordinates, dimensions, aspect ratio, confidence, and extraction outcome even when the thumbnail image is not saved.

## Non-interactive examples

Extract thumbnails automatically:

```bash
python batch_extract.py /path/to/images --recursive --extract-thumbnail --non-interactive
```

Never extract thumbnails:

```bash
python batch_extract.py /path/to/images --recursive --no-extract-thumbnail --non-interactive
```

Choose a Tesseract language:

```bash
python batch_extract.py /path/to/images --ocr-lang eng
```

Use multiple installed Tesseract languages:

```bash
python batch_extract.py /path/to/images --ocr-lang eng+jpn
```

## JSON schema

The output schema begins with:

```json
{
  "schema": "gallery-metadata-extraction/v1"
}
```

Major sections include:

```text
source
output
extraction
detected_region
work
fields
thumbnail
warnings
errors
```

A failed extraction still writes JSON, preserving a machine-readable audit trail instead of failing silently.

## Accuracy and auditability

OCR and layout detection are imperfect by nature. BulkOCR therefore retains raw OCR text, normalized values, image coordinates, status fields, warnings, confidence values, and backend diagnostics wherever possible.

Representative samples should be checked before very large batch runs. Known OCR ambiguities should be normalized only when the visual evidence is sufficiently consistent.