import cv2
import numpy as np

def detect_text_regions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast a bit
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Blackhat helps highlight dark text on light background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Gradient in x-direction helps find text-like strokes
    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    min_val, max_val = np.min(grad_x), np.max(grad_x)

    if max_val - min_val > 0:
        grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype("uint8")
    else:
        grad_x = np.zeros_like(gray)

    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Connect nearby text blobs
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h_img, w_img = frame.shape[:2]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        area = w * h
        aspect_ratio = w / float(h) if h > 0 else 0

        # Filter obvious non-text regions
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
    # For Jetson CSI camera / ArduCam using nvargus
    pipeline = gstreamer_pipeline()
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # If using USB camera instead, comment the line above and use:
    # cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

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
            f"Text Regions: {len(boxes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow("Live Text Detection", output)
        cv2.imshow("Threshold", thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
