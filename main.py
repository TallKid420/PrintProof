import cv2
import pandas as pd
import re
import easyocr

# --- CONFIGURATION ---
EXCEL_FILE = "orders.xlsx"
SENSOR_ID = 0
FLIP_X = True        # Flip horizontally (mirror)
FLIP_Y = True        # Flip vertically (upside down)
USE_GPU = False      # Set True later once GPU is fixed

# --- PREVIEW vs CAPTURE RESOLUTION ---
PREVIEW_WIDTH  = 1280
PREVIEW_HEIGHT = 720
CAPTURE_WIDTH  = 3280
CAPTURE_HEIGHT = 2464
CAPTURE_FPS    = 21

# --- INIT OCR (once at startup) ---
print("🌀 Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=USE_GPU)
print("✅ EasyOCR ready.")

# ─────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────
def gstreamer_pipeline(width=1280, height=720, fps=30):
    return (
        f"nvarguscamerasrc sensor-id={SENSOR_ID} "
        "exposuretimerange=\"10000000 80000000\" "
        "gainrange=\"1 4\" "
        "ispdigitalgainrange=\"1 1\" "
        "wbmode=1 ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
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
# PREVIEW (low res) → CAPTURE (full res)
# ─────────────────────────────────────────
def live_preview_and_capture():
    print("📷 Live preview started.")
    print("   → Press ENTER to capture at full resolution.")
    print("   → Press Q to quit.")

    # Open preview stream
    preview_cap = cv2.VideoCapture(
        gstreamer_pipeline(PREVIEW_WIDTH, PREVIEW_HEIGHT, 30),
        cv2.CAP_GSTREAMER
    )

    if not preview_cap.isOpened():
        raise RuntimeError("Cannot open preview pipeline.")

    capture_frame = None

    while True:
        ret, f = preview_cap.read()
        if not ret or f is None:
            print("⚠️ Frame drop, retrying...")
            continue

        f = apply_flip(f)
        cv2.imshow("PrintProof - ENTER to capture | Q to quit", f)

        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            print("📸 Capturing full resolution image...")
            preview_cap.release()
            cv2.destroyAllWindows()

            # Open a NEW high-res capture
            capture_cap = cv2.VideoCapture(
                gstreamer_pipeline(CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS),
                cv2.CAP_GSTREAMER
            )

            if not capture_cap.isOpened():
                raise RuntimeError("Cannot open full-res capture pipeline.")

            # Discard first few frames (camera needs to settle)
            for _ in range(5):
                capture_cap.read()

            ret2, full_frame = capture_cap.read()
            capture_cap.release()

            if not ret2 or full_frame is None:
                raise RuntimeError("Full-res capture failed.")

            full_frame = apply_flip(full_frame)
            cv2.imwrite("last_capture.jpg", full_frame)
            print("✅ Full-res image captured and saved as last_capture.jpg")
            capture_frame = full_frame
            break

        elif key in (ord('q'), ord('Q')):
            print("❌ Cancelled.")
            preview_cap.release()
            cv2.destroyAllWindows()
            break

    if capture_frame is None:
        raise RuntimeError("No image captured.")

    return capture_frame

# ─────────────────────────────────────────
# OCR
# ─────────────────────────────────────────
def ocr_text(img_bgr):
    # Upscale 2x → helps EasyOCR read small text
    h, w = img_bgr.shape[:2]
    img_big = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    results = reader.readtext(img_big)

    # Print each detected block with confidence
    print("\n--- RAW OCR BLOCKS ---")
    for (bbox, text, conf) in results:
        print(f"  [{conf:.2f}] {text}")
    print("----------------------\n")

    # Combine all text
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

    # --- Try Order ID first ---
    oid = extract_order_id(text_u)
    if oid is not None:
        print(f"🔢 Detected Order ID: {oid}")
        match = df[df["Order ID"] == oid]
        if not match.empty:
            row = match.iloc[0]
            print("✅ PASS: Order ID matched!")
            print(f"   Name:    {row['Name']}")
            print(f"   Title:   {row['Title']}")
            print(f"   Address: {row['Address']}, {row['City']}, {row['State']} {row['ZIP']}")
            return True, row
        else:
            print(f"❌ FAIL: Order ID {oid} not found in spreadsheet.")
            return False, None

    # --- Fallback: Name match ---
    print("⚠️ No Order ID found — trying Name match...")
    for _, row in df.iterrows():
        name = str(row["Name"]).upper()
        if name and name in text_u:
            print(f"✅ PASS: Name matched → {row['Name']}")
            return True, row

    print("❌ FAIL: No Order ID or Name match found.")
    return False, None

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("📂 Loading spreadsheet...")
    df = pd.read_excel(EXCEL_FILE)
    print(f"   Loaded {len(df)} orders.\n")

    # Step 1: Preview + Capture
    frame = live_preview_and_capture()

    # Step 2: OCR
    print("🔍 Running OCR on captured image...")
    text = ocr_text(frame)
    print(f"📝 Full detected text:\n{text}\n")

    # Step 3: Match
    passed, matched_row = match_against_spreadsheet(text, df)

    # Step 4: Final result
    print("\n" + "="*40)
    if passed:
        print("✅  RESULT: PASS")
    else:
        print("❌  RESULT: FAIL")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
