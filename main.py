import cv2
import pandas as pd
import re
import os
import time
import shutil
import subprocess
import easyocr

try:
    import smbus2
    _SMBUS_OK = True
except ImportError:
    _SMBUS_OK = False

# --- CONFIGURATION ---
EXCEL_FILE = "orders.xlsx"
SENSOR_ID = 0
FLIP_X = True
FLIP_Y = True
USE_GPU = False  # Keep False until Jetson PyTorch/CUDA is solved

# Preview/capture settings
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
PREVIEW_FPS = 30

CAPTURE_WIDTH = 3280
CAPTURE_HEIGHT = 2464
CAPTURE_FPS = 21

# Autofocus settings
AUTOFOCUS_ENABLED = True
I2C_BUS = 1        # camera connector on Jetson is bus 1
FOCUS_ADDR = 0x30  # DW9714/DW9807 VCM driver (confirmed via i2cdetect)
AF_STEPS = 16      # number of positions to sweep (higher = slower but more accurate)


print("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=USE_GPU)
print("EasyOCR ready.")


def gstreamer_pipeline(width, height, fps):
    # Brightness-friendly auto behavior (may add noise in low light)
    return (
        f"nvarguscamerasrc sensor-id={SENSOR_ID} "
        "wbmode=1 "
        "intent=3 "
        "exposuretimerange=\"1000000 80000000\" "
        "gainrange=\"1 16\" "
        "ispdigitalgainrange=\"1 8\" ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def apply_flip(frame):
    if FLIP_X and FLIP_Y:
        return cv2.flip(frame, -1)
    if FLIP_X:
        return cv2.flip(frame, 1)
    if FLIP_Y:
        return cv2.flip(frame, 0)
    return frame


def open_camera(width, height, fps, retries=5, retry_sleep=0.4):
    pipeline = gstreamer_pipeline(width, height, fps)
    last_cap = None
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        last_cap = cap
        if cap.isOpened():
            return cap
        cap.release()
        print(f"Camera open attempt {attempt}/{retries} failed")
        time.sleep(retry_sleep)

    if last_cap is not None:
        last_cap.release()

    raise RuntimeError(
        "Cannot open camera pipeline.\n"
        f"Pipeline: {pipeline}\n"
        "Common causes:\n"
        " - pip-installed opencv overriding system opencv (must use apt python3-opencv)\n"
        " - another process is using the camera\n"
        " - nvargus-daemon is unhealthy (try: sudo systemctl restart nvargus-daemon)\n"
    )


def _have_cmd(cmd):
    return shutil.which(cmd) is not None


def _vcm_set_position(pos):
    """
    Write a 10-bit position (0-1023) to DW9714/DW9807 VCM at FOCUS_ADDR.
    0 = infinity (far), 1023 = macro (close).
    Tries smbus2 first (no sudo needed), falls back to i2cset subprocess.
    """
    pos = max(0, min(1023, pos))
    byte0 = (pos >> 4) & 0x3F
    byte1 = (pos & 0x0F) << 4

    if _SMBUS_OK:
        try:
            with smbus2.SMBus(I2C_BUS) as bus:
                bus.write_byte_data(FOCUS_ADDR, byte0, byte1)
            return
        except Exception as e:
            print(f"  smbus2 write failed (pos={pos}): {e}")

    # Fallback: subprocess i2cset with full error capture
    cmd = ["sudo", "i2cset", "-y", str(I2C_BUS),
           f"0x{FOCUS_ADDR:02x}", f"0x{byte0:02x}", f"0x{byte1:02x}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  i2cset write failed (pos={pos}): {result.stderr.strip()}")


def _sharpness(frame):
    """Laplacian variance: higher = sharper/more in-focus."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def trigger_autofocus(preview_cap=None):
    """
    Sweep VCM from far (0) to near (1023) in AF_STEPS steps.
    Score each frame for sharpness, then lock to the best position.
    Pass preview_cap so we can read live frames during the sweep.
    """
    if not AUTOFOCUS_ENABLED:
        return

    if not _have_cmd("i2cset"):
        print("Autofocus skipped: i2cset not installed (sudo apt install i2c-tools)")
        return

    print(f"Autofocus: sweeping lens ({AF_STEPS} steps)...")

    step_size = 1024 // AF_STEPS
    positions = list(range(0, 1024, step_size))
    best_pos = positions[0]
    best_score = -1.0

    for pos in positions:
        _vcm_set_position(pos)
        time.sleep(0.08)  # let motor settle

        if preview_cap is not None and preview_cap.isOpened():
            ret, frame = preview_cap.read()
            if ret and frame is not None:
                score = _sharpness(frame)
                print(f"  pos={pos:4d}  sharpness={score:.1f}")
                if score > best_score:
                    best_score = score
                    best_pos = pos

    print(f"Autofocus: locked at position={best_pos} (sharpness={best_score:.1f})")
    _vcm_set_position(best_pos)
    time.sleep(0.15)


def live_preview_and_capture():
    print("Live preview started")
    print("  Enter: capture full resolution")
    print("  q: quit")

    cap = open_camera(PREVIEW_WIDTH, PREVIEW_HEIGHT, PREVIEW_FPS)
    captured = None

    while True:
        ret, f = cap.read()
        if not ret or f is None:
            print("Frame drop, retrying...")
            continue

        f = apply_flip(f)
        cv2.imshow("PrintProof Preview (Enter=capture, q=quit)", f)

        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter
            print("Autofocusing...")
            trigger_autofocus(preview_cap=cap)  # sweep while preview stream is still open

            cap.release()
            cv2.destroyAllWindows()

            print("Capturing full resolution image...")
            cap_hires = open_camera(CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS)

            # Let exposure settle on the new stream
            for _ in range(6):
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

        if key in (ord("q"), ord("Q")):
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("User cancelled capture")

    return captured


def ocr_text(img_bgr):
    # Upscale helps OCR for small text
    h, w = img_bgr.shape[:2]
    img_big = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    results = reader.readtext(img_big)

    print("\nRAW OCR BLOCKS:")
    for (_, txt, conf) in results:
        print(f"  [{conf:.2f}] {txt}")
    print("")

    full_text = " ".join([r[1] for r in results])
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return full_text


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
            print("PASS: Order ID matched")
            print(f"  Name:    {row['Name']}")
            print(f"  Title:   {row['Title']}")
            print(f"  Address: {row['Address']}, {row['City']}, {row['State']} {row['ZIP']}")
            return True, row
        print(f"FAIL: Order ID {oid} not found in spreadsheet")
        return False, None

    print("No Order ID found; trying Name match...")
    for _, row in df.iterrows():
        name = str(row["Name"]).upper()
        if name and name in text_u:
            print(f"PASS: Name matched -> {row['Name']}")
            return True, row

    print("FAIL: No Order ID or Name match found")
    return False, None


def main():
    print("Loading spreadsheet...")
    df = pd.read_excel(EXCEL_FILE)
    print(f"Loaded {len(df)} orders")

    frame = live_preview_and_capture()

    print("Running OCR on captured image...")
    text = ocr_text(frame)
    print(f"Full detected text:\n{text}\n")

    passed, _ = match_against_spreadsheet(text, df)

    print("=" * 40)
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    print("=" * 40)


if __name__ == "__main__":
    main()
