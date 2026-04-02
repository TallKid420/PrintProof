import cv2
import pandas as pd
import re
import pytesseract

EXCEL_FILE = "orders.xlsx"
SENSOR_ID = 0

def gstreamer_pipeline(sensor_id=0, width=1280, height=720, fps=30):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
    )

def capture_frame():
    cap = cv2.VideoCapture(gstreamer_pipeline(SENSOR_ID), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open camera pipeline (CAP_GSTREAMER).")

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Camera capture failed (read() returned no frame).")
    return frame

def preprocess_for_ocr(img_bgr):
    # Helps plaques/labels: grayscale + denoise + threshold
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
    config = r"--oem 1 --psm 6"  # good general setting for blocks of text
    text = pytesseract.image_to_string(proc, config=config)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_order_id(text):
    # Adjust if your Order IDs have different format
    m = re.search(r"\b(100\d|10\d\d|\d{4})\b", text)
    return int(m.group(0)) if m else None

def main():
    print("Loading spreadsheet...")
    df = pd.read_excel(EXCEL_FILE)

    print("Capturing image...")
    frame = capture_frame()

    print("Running OCR (CPU)...")
    text = ocr_text(frame)
    text_u = text.upper()
    print(f"\n--- OCR TEXT ---\n{text}\n---------------\n")

    oid = extract_order_id(text_u)
    if oid is not None:
        print(f"Found Order ID: {oid}")
        match = df[df["Order ID"] == oid]
        if match.empty:
            print("FAIL: Order ID not found in spreadsheet")
            return
        row = match.iloc[0]
        print("PASS: Order found")
        print(f"Expected Name: {row['Name']}")
        print(f"Expected Title: {row['Title']}")
        # Add more field checks here if you want
        return

    # Fallback: name match
    for _, row in df.iterrows():
        name = str(row["Name"]).upper()
        if name and name in text_u:
            print(f"PASS: Name match found: {row['Name']}")
            return

    print("FAIL: No Order ID or Name match found")

if __name__ == "__main__":
    main()
