import cv2
import easyocr
import pandas as pd
import re

EXCEL_FILE = 'orders.xlsx'
SENSOR_ID = 0

print("Loading OCR...")
reader = easyocr.Reader(['en'], gpu=True)

# Load spreadsheet once
df = pd.read_excel(EXCEL_FILE)

def get_pipeline():
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
    )

def extract_order_id(text):
    # Looks for 4-digit numbers like 1001, 1002
    match = re.search(r'\b\d{4}\b', text)
    return match.group(0) if match else None

def check_plaque():
    cap = cv2.VideoCapture(get_pipeline(), cv2.CAP_GSTREAMER)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Camera capture failed")
        return

    print("Running OCR...")
    results = reader.readtext(frame)
    detected_text = " ".join([r[1] for r in results]).upper()

    print(f"Detected:\n{detected_text}\n")

    # --- Try Order ID first ---
    order_id = extract_order_id(detected_text)

    if order_id:
        print(f"Found Order ID: {order_id}")
        match = df[df['Order ID'] == int(order_id)]

        if not match.empty:
            row = match.iloc[0]
            print("PASS: Order found in spreadsheet")

            # Optional deeper checks
            name = row['Name'].upper()
            if name in detected_text:
                print("Name matches")
            else:
                print("Name mismatch")

        else:
            print("FAIL: Order ID not found")

    else:
        print("No Order ID found — trying Name match")

        # fallback: match by name
        for _, row in df.iterrows():
            name = str(row['Name']).upper()
            if name in detected_text:
                print(f"PASS: Found match for {name}")
                return

        print("FAIL: No match found")

if __name__ == "__main__":
    check_plaque()
