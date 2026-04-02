import cv2
import numpy as np
import os
import time

I2C_BUS = 4
VCM_ADDR = 0x0c

def set_focus(value):
    value = max(0, min(1023, int(value)))
    packed = (value << 4) & 0x3FF0
    data1 = (packed >> 8) & 0x3F
    data2 = packed & 0xF0
    cmd = f"i2cset -y {I2C_BUS} 0x{VCM_ADDR:02x} {data1} {data2}"
    return os.system(cmd)

def sharpness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def autofocus(cap, focus_min=50, focus_max=900, step=50):
    best_focus = focus_min
    best_score = -1

    print("Running autofocus...")

    for focus in range(focus_min, focus_max + 1, step):
        set_focus(focus)
        time.sleep(0.2)

        # grab a couple frames so the camera settles
        for _ in range(2):
            cap.read()

        ret, frame = cap.read()
        if not ret:
            continue

        score = sharpness_score(frame)
        print(f"focus={focus}, score={score:.2f}")

        if score > best_score:
            best_score = score
            best_focus = focus

    set_focus(best_focus)
    print(f"Best focus: {best_focus}, score={best_score:.2f}")
    return best_focus, best_score

def detect_text_regions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)

    min_val, max_val = np.min(grad_x), np.max(grad_x)
    if max_val - min_val > 0:
        grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype("uint8")
    else:
        grad_x = np.zeros_like(gray)

    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h_img, w_img = frame.shape[:2]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect_ratio = w / float(h) if h > 0 else 0

        if area < 300:
            continue
        if w < 20 or h < 8:
            continue
        if aspect_ratio < 1.2:
            continue
        if w > w_img * 0.95 or h > h_img * 0.95:
            continue

        boxes.append((x, y, w, h))

    return boxes, thresh

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

def main():
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    # For USB camera instead:
    # cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    best_focus, best_score = autofocus(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        boxes, thresh = detect_text_regions(frame)
        output = frame.copy()

        for (x, y, w, h) in boxes:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            output,
            f"Focus: {best_focus}  Score: {best_score:.2f}  Text Regions: {len(boxes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            "Press 'a' to autofocus, 'q' or ESC to quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow("Autofocus + Live Text Detection", output)
        cv2.imshow("Threshold", thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('a'):
            best_focus, best_score = autofocus(cap)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
