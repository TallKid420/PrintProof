# PrintProof

PrintProof is a Jetson-focused camera inspection workflow with two modes:

- `ocr/`: OCR-focused pipeline
- `Iris/`: LLM-first pipeline with an interactive terminal and batch-aware order matching

## Repository Layout

- `Iris/`
- `ocr/`
- `orders.xlsx`
- `requirements.txt`

## System Requirements (Jetson)

Install system packages:

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv
```

If you plan to use OCR mode, also install Tesseract:

```bash
sudo apt install -y tesseract-ocr
```

## Python Dependencies

Install from the project root:

```bash
pip3 install --user -r requirements.txt
```

## Iris Mode (LLM + Batch Terminal)

Run:

```bash
cd Iris
python3 main.py
```

This opens the `iris>` command terminal.

### Iris Commands

- `help`
- `list`
- `batch <orders.xlsx|orders.json>`
- `run <image_path>`
- `runid <order_id> <image_path>`
- `rerun`
- `camera`
- `reload`
- `quit`

### Batch and JSON Behavior

- `batch <file.xlsx>` processes that workbook into active order data
- Active orders are written to `Iris/Json/orders.json`
- `batch <file.json>` loads an existing batch JSON from `Iris/batch`

### Logging

Iris logs command and capture activity to:

- `Iris/logs/iris.log`

## OCR Mode

Run:

```bash
cd ocr
python3 main.py
```

OCR mode does not use the LLM module.

## Notes

- Camera capture uses OpenCV + GStreamer on Jetson.
- LLM requests in Iris mode are sent to local Ollama (`gemma3:4b`) at `http://127.0.0.1:11434`.
