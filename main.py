import cv2
import pandas as pd
import re
import os
import time
import shutil
import easyocr

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
I2C_BUS = 1                  # usually 1 on Jetson camera connector
FOCUS_I2C_ADDR_CANDIDATES = [0x0c, 0x0d, 0x18]  # common candidates for Arducam focus boards
AUTOFOCUS_SLEEP_SEC = 1.5    # give motor time to move


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


def _i2c_addr_present(bus, addr_hex):
    # Returns True if i2cdetect shows a device at addr
    # Requires i2c-tools installed.
    try:
        out = os.popen(f"i2cdetect -y {bus}").read()
        # i2cdetect prints address in hex without 0x prefix (e.g. "0c")
        return f"{addr_hex:02x}" in out.lower()
    except Exception:
        return False


def find_focus_addr():
    # Best effort: if i2c-tools exist, scan for known candidates
    if not _have_cmd("i2cdetect"):
        return None

    for addr in FOCUS_I2C_ADDR_CANDIDATES:
        if _i2c_addr_present(I2C_BUS, addr):
            return addr
    return None


def trigger_autofocus():
    if not AUTOFOCUS_ENABLED:
        return

    if not _have_cmd("i2cset"):
        print("Autofocus skipped: i2cset not installed (sudo apt install i2c-tools)")
        return

    addr = find_focus_addr()
    if addr is None:
        # Fall back to 0x0c if we cannot detect
        addr = 0x0c

    # NOTE: This sequence is device-specific. If your board ignores it,
    # we can switch to a different register sequence once you confirm the focus module model.
    print(f"Autofocus: triggering focus motor on i2c bus {I2C_BUS}, addr 0x{addr:02x}")
    rc = os.system(f"i2cset -y {I2C_BUS} 0x{addr:02x} 0x01 0x00")
    if rc != 0:
        print("Autofocus command returned non-zero (may not be supported on this focus module).")
    time.sleep(AUTOFOCUS_SLEEP_SEC)


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
            cap.release()
            cv2.destroyAllWindows()

            # Trigger autofocus right before high-res capture
            trigger_autofocus()

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
