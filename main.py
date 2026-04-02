import cv2
import pandas as pd
import re
import os
import time
import shutil
import subprocess
import base64
import requests

try:
    import smbus2
    _SMBUS_OK = True
except ImportError:
    _SMBUS_OK = False

# --- CONFIGURATION ---
EXCEL_FILE   = "orders.xlsx"
SENSOR_ID    = 0
FLIP_X       = True
FLIP_Y       = True

# Preview/capture settings
PREVIEW_WIDTH  = 1280
PREVIEW_HEIGHT = 720
PREVIEW_FPS    = 30

CAPTURE_WIDTH  = 3280
CAPTURE_HEIGHT = 2464
CAPTURE_FPS    = 21

# Autofocus settings
AUTOFOCUS_ENABLED = True
I2C_BUS    = 1
FOCUS_ADDR = 0x30
AF_STEPS   = 16

# Ollama settings
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava:7b"
CAPTURE_PATH = "last_capture.jpg"


# ─────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────
def gstreamer_pipeline(width, height, fps):
    return (
        f"nvarguscamerasrc sensor-id={SENSOR_ID} "
        "wbmode=1 intent=3 "
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
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()
        print(f"Camera open attempt {attempt}/{retries} failed")
        time.sleep(retry_sleep)
    raise RuntimeError(
        f"Cannot open camera after {retries} attempts.\n"
        "Check: camera connected? nvargus-daemon running? Another process using camera?"
    )


# ─────────────────────────────────────────
# AUTOFOCUS (VCM sweep)
# ─────────────────────────────────────────
def _have_cmd(cmd):
    return shutil.which(cmd) is not None


def _vcm_set_position(pos):
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

    cmd = ["sudo", "i2cset", "-y", str(I2C_BUS),
           f"0x{FOCUS_ADDR:02x}", f"0x{byte0:02x}", f"0x{byte1:02x}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  i2cset write failed (pos={pos}): {result.stderr.strip()}")


def _sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def trigger_autofocus(preview_cap=None):
    if not AUTOFOCUS_ENABLED:
        return
    if not (_SMBUS_OK or _have_cmd("i2cset")):
        print("Autofocus skipped: no smbus2 or i2cset available")
        return

    print(f"Autofocus: sweeping lens ({AF_STEPS} steps)...")
    step_size = 1024 // AF_STEPS
    positions  = list(range(0, 1024, step_size))
    best_pos   = positions[0]
    best_score = -1.0

    for pos in positions:
        _vcm_set_position(pos)
        time.sleep(0.08)
        if preview_cap is not None and preview_cap.isOpened():
            ret, frame = preview_cap.read()
            if ret and frame is not None:
                score = _sharpness(frame)
                print(f"  pos={pos:4d}  sharpness={score:.1f}")
                if score > best_score:
                    best_score = score
                    best_pos   = pos

    print(f"Autofocus: locked at pos={best_pos} (sharpness={best_score:.1f})")
    _vcm_set_position(best_pos)
    time.sleep(0.15)


# ─────────────────────────────────────────
# PREVIEW + CAPTURE
# ─────────────────────────────────────────
def live_preview_and_capture():
    print("Live preview started — ENTER to capture, Q to quit")
    cap = open_camera(PREVIEW_WIDTH, PREVIEW_HEIGHT, PREVIEW_FPS)

    while True:
        ret, f = cap.read()
        if not ret or f is None:
            continue

        cv2.imshow("PrintProof (ENTER=capture, Q=quit)", apply_flip(f))
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter
            print("Autofocusing...")
            trigger_autofocus(preview_cap=cap)

            cap.release()
            cv2.destroyAllWindows()

            print("Capturing full resolution...")
            cap_hi = open_camera(CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS)
            for _ in range(6):
                cap_hi.read()
            ret2, frame = cap_hi.read()
            cap_hi.release()

            if not ret2 or frame is None:
                raise RuntimeError("Full-res capture failed.")

            frame = apply_flip(frame)
            cv2.imwrite(CAPTURE_PATH, frame)
            print(f"Saved: {CAPTURE_PATH}")
            return frame

        if key in (ord("q"), ord("Q")):
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("User cancelled.")


# ─────────────────────────────────────────
# OLLAMA LLAVA ANALYSIS
# ─────────────────────────────────────────
def analyze_with_llava(image_path):
    """Send image to local Ollama LLaVA and extract plaque text."""
    print(f"Sending image to {OLLAMA_MODEL} via Ollama...")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "This is a photo of a printed plaque or label. "
        "Please read ALL text visible on it carefully. "
        "Return ONLY the raw text you see, exactly as printed — "
        "including any order number, name, title, and address. "
        "Do not add commentary or formatting."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        print(f"\nLLaVA response:\n{text}\n")
        return text
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Is it running?\n"
            "Start it with: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")


# ─────────────────────────────────────────
# MATCHING
# ─────────────────────────────────────────
def extract_order_id(text):
    m = re.search(r"\b(\d{4})\b", text)
    return int(m.group(1)) if m else None


def match_against_spreadsheet(text, df):
    text_u = text.upper()

    # Try Order ID first
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
        print(f"FAIL: Order ID {oid} not found in spreadsheet.")
        return False, None

    # Fallback: name match
    print("No Order ID found — trying Name match...")
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

    # Capture image
    live_preview_and_capture()

    # Analyze with LLaVA
    text = analyze_with_llava(CAPTURE_PATH)

    # Match against spreadsheet
    passed, matched_row = match_against_spreadsheet(text, df)

    print("\n" + "=" * 40)
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
