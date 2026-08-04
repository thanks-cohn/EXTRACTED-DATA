# EXTRACTED-DATA

A foundational screenshot-analysis tool for locating a gallery metadata block inside a much larger image and then extracting structured data from that validated block.

The program does **not** begin by OCR-reading the whole screenshot and guessing what matters. It uses a strict two-stage process:

1. **Find and validate the metadata block.**
2. **Apply extraction logic only inside that block.**

The source image is never modified.

## Install first

### Linux

```bash
cd ~/dev
git clone git@github.com:thanks-cohn/EXTRACTED-DATA.git
cd EXTRACTED-DATA
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Tesseract OCR through your system package manager.

Arch/Fedora/Debian examples:

```bash
# Arch Linux
sudo pacman -S tesseract tesseract-data-eng

# Fedora
sudo dnf install tesseract tesseract-langpack-eng

# Debian / Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-eng
```

Confirm it works:

```bash
tesseract --version
```

## Run it

```bash
python extracted_data.py /path/to/screenshot.png
```

The program analyzes the image, asks whether to extract the detected thumbnail, and always writes JSON beside the source image.

Example source:

```text
/path/to/example.png
```

Required JSON output:

```text
/path/to/example-EXTRACTED-DATA.json
```

That naming rule is fixed:

```text
<original-image-name-without-extension>-EXTRACTED-DATA.json
```

## Thumbnail prompts

When a thumbnail is detected, interactive mode asks:

```text
Extract thumbnail? [y/N]:
```

- `y` extracts it.
- `n` or Enter leaves it unextracted.

When extraction is selected, it asks:

```text
Use default location or choose a new one? [D/n]:
```

- `d` or Enter saves beside the source image.
- `n` asks for another directory.

The default thumbnail filename is:

```text
<original-image-name-without-extension>-THUMBNAIL.png
```

The JSON records thumbnail bounds, width, height, aspect ratio, confidence, validation evidence, and extraction outcome even when the thumbnail itself is not saved.

## Non-interactive use

Extract the thumbnail automatically:

```bash
python extracted_data.py screenshot.png --extract-thumbnail --non-interactive
```

Never extract it:

```bash
python extracted_data.py screenshot.png --no-extract-thumbnail --non-interactive
```

Choose an OCR language:

```bash
python extracted_data.py screenshot.png --ocr-lang eng
```

Multiple installed Tesseract languages can be combined:

```bash
python extracted_data.py screenshot.png --ocr-lang eng+jpn
```

## Stage 1: detect the block

The foundational detector looks for the stable label stack:

```text
Group
Type
Language
Series
Characters
Tags
```

It does not depend on one fixed location in the screenshot. It searches for a vertically ordered, left-aligned cluster of those labels.

`Group` and `Tags` are mandatory anchors. Other labels strengthen confidence.

After finding the stack, the program expands upward to include up to two broad header layers immediately above `Group`:

```text
[ title bar ]
[ creator / author bar ]
Group
Type
Language
Series
Characters
Tags
```

It extends downward through wrapped tag rows and stops after the final active row or blank boundary.

Everything else in the larger screenshot is ignored during field extraction.

If the block cannot be found confidently, the program does not blindly extract unrelated text. It still creates the required JSON beside the image, but records a failed extraction status.

## Stage 2: extract inside the validated block

Only after block detection succeeds does the program extract:

- title;
- creator or author line;
- date on the far right;
- Group;
- Type;
- Language;
- Series;
- Characters;
- Tags;
- detected metadata-region coordinates;
- OCR anchors;
- header-band coordinates;
- thumbnail evidence and dimensions;
- optional thumbnail extraction information;
- warnings and errors.

Characters and tags are detected as compact, filled text blocks. The detector does not calibrate one exact shade of gray. It accepts compact non-white filled blocks and rejects plain white surroundings such as speech bubbles.

Characters are read only from the Characters section. Tags are read only from the Tags section, including wrapped rows.

## Thumbnail logic

Thumbnail detection is separate from metadata extraction.

The program searches immediately to the left of the validated metadata block for an image-like rectangular region. It strengthens the match by looking beneath the candidate for the recurring controls:

```text
Read Online
Download
```

This relationship identifies the thumbnail more reliably than rectangle detection alone.

The JSON always notes the detected thumbnail size, for example:

```json
{
  "thumbnail": {
    "status": "detected",
    "size": {
      "width_px": 300,
      "height_px": 200
    },
    "aspect_ratio": 1.5,
    "extraction": {
      "requested": false,
      "performed": false,
      "output_path": null
    }
  }
}
```

## JSON structure

The output schema begins with:

```json
{
  "schema": "gallery-metadata-extraction/v1"
}
```

A successful record includes these major sections:

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

`fields` contains:

```text
group
type
language
series
characters
tags
```

Character and tag entries include both normalized values and detected image coordinates.

## Failure output

A failure still produces:

```text
<image-name>-EXTRACTED-DATA.json
```

Example:

```json
{
  "extraction": {
    "status": "failed",
    "reason": "metadata_label_cluster_not_found"
  },
  "detected_region": {
    "status": "not-found",
    "box": null
  }
}
```

This preserves a machine-readable audit trail and prevents silent failures.

## Current status

This is the first foundational implementation. Its primary responsibility is to establish the correct descriptive block inside a larger screenshot before applying any deeper extraction logic.

OCR and layout detection are imperfect by nature. Results should be checked against representative screenshots before large batch runs. The JSON retains coordinates and status information so later revisions can be audited and improved.
