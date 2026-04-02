# PrintProof

AI-powered plaque inspection system running on NVIDIA Jetson Orin Nano Super.
Captures an image via ArduCam IMX219, runs OCR, and validates the print
against an orders spreadsheet.

---

## Hardware
- NVIDIA Jetson Orin Nano Super (JetPack 6.2 / L4T R36.4.7)
- ArduCam IMX219 (8MP)

---

## Setup

### 1. System dependencies
```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv tesseract-ocr
