import cv2
import pandas as pd
import re
import easyocr
import numpy as np

# --- CONFIGURATION ---
EXCEL_FILE   = "orders.xlsx"
SENSOR_ID    = 0
FLIP_X       = True   # Flip horizontally (mirror)
FLIP_Y       = True   # Flip vertically (upside down)
USE_GPU      = False  # Set True once GPU/PyTorch is fixed

# --- RESOLUTION ---
PREVIEW_WIDTH   = 1280
PREVIEW_HEIGHT  = 720
CAPTURE_WIDTH   = 3280
CAPTURE_HEIGHT  = 2464
CAPTURE_FPS     = 21

# --- INIT OCR (once at startup) ---
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=USE_GPU)
print("EasyOCR ready.")

# ─────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────
def gstreamer_pipeline(width, height, fps):
    return (
        f"nvarguscamerasrc sensor-id={SENSOR_ID} "
        "exposuretimerange=\"10000000 80000000\" "
        "gainrange=\"1 4\" "
        "ispdigitalgainrange=\"1 1\" "
        "wbmode=1 ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )

# ─────────────────────────────────────────
# FLIP
# ─────────────────────────────────────────
def apply_flip(frame):
    if FLIP_X and FLIP_Y:
        return cv2.flip(frame, -1)
    elif FLIP_X:
        return cv2.flip(frame, 1)
    elif FLIP_Y:
        return cv2.flip(frame, 0)
    return frame

# ─────────────────────────────────────────
# OPEN CAMERA (with retries)
# ─────────────────────────────────────────
def open_camera(width, height, fps, retries=3):
    pipeline = gstreamer_pipeline(width, height, fps)
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()
        print(f"Camera open attempt {attempt}/{retries} failed, retrying...")
    raise RuntimeError(
        f"Cannot open camera pipeline after {retries} attempts.\n"
        f"Pipeline: {pipeline}\n"
        "Check: camera connected? nvarguscamerasrc available? Another process using camera?"
    )

# ─────────────────────────────────────────
# PREVIEW + CAPTURE
# ─────────────────────────────────────────
def live_preview_and_capture():
    print("Starting live preview...")
    print("  Press ENTER to capture at full resolution.")
    print("  Press Q to quit.")

    cap = open_camera(PREVIEW_WIDTH, PREVIEW_HEIGHT, 30)
    captured = None

    while True:
        ret, f = cap.read()
        if not ret or f is None:
            print("Frame drop, retrying...")
            continue

        f = apply_flip(f)
        cv2.imshow("PrintProof - ENTER to capture | Q to quit", f)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            cap.release()
            cv2.destroyAllWindows()
            print("Capturing full resolution image...")

            cap_hires = open_camera(CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS)

            # Discard first few frames so camera settles
            for _ in range(5):
                cap_hires.read()

            ret2, full_frame = cap_hires.read()
            cap_hires.release()

            if not ret2 or full_frame is None:
                raise RuntimeError("Full-res capture failed.")

            full_frame = apply_flip(full_frame)
            cv2.imwrite("last_capture.jpg", full_frame)
            print("Captured and saved as last_capture.jpg")
            captured = full_frame
            break

        elif key in (ord('q'), ord('Q')):
            print("Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            break

    if captured is None:
        raise RuntimeError("No image captured.")
    return captured

# ─────────────────────────────────────────
# OCR
# ─────────────────────────────────────────
def ocr_text(img_bgr):
    h, w = img_bgr.shape[:2]
    img_big = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    results = reader.readtext(img_big)

    print("\n--- RAW OCR BLOCKS ---")
    for (bbox, text, conf) in results:
        print(f"  [{conf:.2f}] {text}")
    print("----------------------\n")

    full_text = " ".join([res[1] for res in results])
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return full_text

# ─────────────────────────────────────────
# MATCHING
# ─────────────────────────────────────────
def extract_order_id(text):
    m = re.search(r"\b\d{4}\b", text)
    return int(m.group(0)) if m else None

def match_against_spreadsheet(text, df):
    text_u = text.upper()

    oid = extract_order_id(text_u)
    if oid is not None:
        print(f"Detected Order ID: {oid}")
        match = df[df["Order ID"] == oid]
        if not match.empty:
            row = match.iloc[0]
            print("PASS: Order ID matched!")
            print(f"  Name:    {row['Name']}")
            print(f"  Title:   {row['Title']}")
            print(f"  Address: {row['Address']}, {row['City']}, {row['State']} {row['ZIP']}")
            return True, row
        else:
            print(f"FAIL: Order ID {oid} not found in spreadsheet.")
            return False, None

    print("No Order ID found -- trying Name match...")
    for _, row in df.iterrows():
        name = str(row["Name"]).upper()
        if name and name in text_u:
            print(f"PASS: Name matched -> {row['Name']}")
            return True, row

    print("FAIL: No Order ID or Name match found.")
    return False, None

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("Loading spreadsheet...")
    df = pd.read_excel(EXCEL_FILE)
    print(f"Loaded {len(df)} orders.\n")

    frame = live_preview_and_capture()

    print("Running OCR on captured image...")
    text = ocr_text(frame)
    print(f"Full detected text:\n{text}\n")

    passed, matched_row = match_against_spreadsheet(text, df)

    print("\n" + "=" * 40)
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
