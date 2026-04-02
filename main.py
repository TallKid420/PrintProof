import cv2
import pandas as pd
import re
import pytesseract

# --- CONFIGURATION ---
EXCEL_FILE = "orders.xlsx"
SENSOR_ID = 0
FLIP_X = False   # Set True to flip horizontally (mirror)
FLIP_Y = False   # Set True to flip vertically (upside down)

def gstreamer_pipeline(sensor_id=0, width=3280, height=2464, fps=21):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
    )

def apply_flip(frame):
    if FLIP_X and FLIP_Y:
        return cv2.flip(frame, -1)   # Both axes
    elif FLIP_X:
        return cv2.flip(frame, 1)    # Horizontal
    elif FLIP_Y:
        return cv2.flip(frame, 0)    # Vertical
    return frame

def live_preview_and_capture():
    print("📷 Starting live preview... Press ENTER to capture, Q to quit.")
    cap = cv2.VideoCapture(gstreamer_pipeline(SENSOR_ID), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open camera pipeline (CAP_GSTREAMER).")

    frame = None
    while True:
        ret, f = cap.read()
        if not ret or f is None:
            print("⚠️ Frame drop, retrying...")
            continue

        f = apply_flip(f)
        cv2.imshow("PrintProof Preview - Press ENTER to capture, Q to quit", f)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:        # ENTER key
            frame = f.copy()
            print("✅ Image captured!")
            break
        elif key == ord('q') or key == ord('Q'):
            print("❌ Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame is None:
        raise RuntimeError("No image captured.")
    return frame

def preprocess_for_ocr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )
    return thr

def ocr_text(img_bgr):
    proc = preprocess_for_ocr(img_bgr)
    config = r"--oem 1 --psm 6"
    text = pytesseract.image_to_string(proc, config=config)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_order_id(text):
    m = re.search(r"\b(100\d|10\d\d|\d{4})\b", text)
    return int(m.group(0)) if m else None

def main():
    print("Loading spreadsheet...")
    df = pd.read_excel(EXCEL_FILE)

    # Live preview → capture on ENTER
    frame = live_preview_and_capture()

    print("🔍 Running OCR...")
    text = ocr_text(frame)
    text_u = text.upper()
    print(f"\n--- OCR TEXT ---\n{text}\n---------------\n")

    # --- Order ID Match ---
    oid = extract_order_id(text_u)
    if oid is not None:
        print(f"🔢 Found Order ID: {oid}")
        match = df[df["Order ID"] == oid]
        if match.empty:
            print("❌ FAIL: Order ID not found in spreadsheet")
            return
        row = match.iloc[0]
        print(f"✅ PASS: Order found!")
        print(f"   Name:    {row['Name']}")
        print(f"   Title:   {row['Title']}")
        print(f"   Address: {row['Address']}, {row['City']}, {row['State']} {row['ZIP']}")
        return

    # --- Fallback: Name Match ---
    for _, row in df.iterrows():
        name = str(row["Name"]).upper()
        if name and name in text_u:
            print(f"✅ PASS: Name match → {row['Name']}")
            return

    print("❌ FAIL: No Order ID or Name match found")

if __name__ == "__main__":
    main()
