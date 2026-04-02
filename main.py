import cv2, easyocr, time
import pandas as pd

# --- CONFIGURATION ---
EXCEL_FILE = 'production_list.xlsx'
COLUMN_NAME = 'PlaqueText'  # Change this to match your spreadsheet column
SENSOR_ID = 0

# Initialize OCR (English, using GPU)
print("Loading AI Models (GPU)...")
reader = easyocr.Reader(['en'], gpu=True)

# GStreamer Pipeline for high-res IMX219
def get_pipeline():
    return (
        f"nvarguscamerasrc sensor-id={SENSOR_ID} ! "
        "video/x-raw(memory:NVMM), width=3280, height=2464, framerate=21/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )

def capture_and_check():
    # 1. Load Spreadsheet
    df = pd.read_excel(EXCEL_FILE)
    
    # 2. Capture Image
    cap = cv2.VideoCapture(get_pipeline(), cv2.CAP_GSTREAMER)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to capture image")
        return

    # 3. Run OCR
    print("Scanning Plaque...")
    results = reader.readtext(frame)
    detected_text = " ".join([res[1] for res in results]).upper()
    print(f"Detected: {detected_text}")

    # 4. Match Logic
    # We check if any row in the spreadsheet is contained within the detected text
    match = df[df[COLUMN_NAME].apply(lambda x: str(x).upper() in detected_text)]

    if not match.empty:
        print("PASS: Match found in spreadsheet!")
        # Here you can add the UPS label check logic next
    else:
        print("FAIL: No match found!")

if __name__ == "__main__":
    capture_and_check()
