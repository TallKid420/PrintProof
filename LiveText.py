# MIT License
# Copyright (c) 2019 JetsonHacks (modified for PrintProof text detection)
# Autofocus base from JetsonHacks, text detection added for plaque inspection

import cv2
import numpy as np
import os
import argparse
from Focuser import Focuser

focuser = None

def focusing(val):
    focuser.set(Focuser.OPT_FOCUS, val)

def laplacian(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_lap = cv2.Laplacian(img_gray, cv2.CV_16U)
    return cv2.mean(img_lap)[0]

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

def gstreamer_pipeline(capture_width=1280, capture_height=720,
                        display_width=640, display_height=360,
                        framerate=60, flip_method=0):
    return (
        'nvarguscamerasrc ! '
        'video/x-raw(memory:NVMM), '
        'width=(int)%d, height=(int)%d, '
        'format=(string)NV12, framerate=(fraction)%d/1 ! '
        'nvvidconv flip-method=%d ! '
        'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
        'videoconvert ! '
        'video/x-raw, format=(string)BGR ! appsink'
        % (capture_width, capture_height, framerate, flip_method,
           display_width, display_height)
    )

def show_camera():
    max_index = 10
    max_value = 0.0
    last_value = 0.0
    dec_count = 0
    focal_distance = 10
    focus_finished = False
    text_detection_on = True

    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    focusing(focal_distance)
    skip_frame = 2

    if cap.isOpened():
        cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)

        while cv2.getWindowProperty('CSI Camera', 0) >= 0:
            ret_val, img = cap.read()
            if not ret_val:
                break

            display = img.copy()

            # --- Text Detection ---
            if focus_finished and text_detection_on:
                boxes, thresh = detect_text_regions(img)
                for (x, y, w, h) in boxes:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display,
                    f"Text Regions: {len(boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Threshold', thresh)
            elif not focus_finished:
                cv2.putText(display,
                    f"Focusing... pos={focal_distance}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(display,
                "ESC=quit  ENTER=refocus  T=toggle text detection",
                (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow('CSI Camera', display)

            # --- Autofocus logic (unchanged from your original) ---
            if skip_frame == 0:
                skip_frame = 3
                if dec_count < 6 and focal_distance < 1000:
                    focusing(focal_distance)
                    val = laplacian(img)
                    if val > max_value:
                        max_index = focal_distance
                        max_value = val
                    if val < last_value:
                        dec_count += 1
                    else:
                        dec_count = 0
                    if dec_count < 6:
                        last_value = val
                        focal_distance += 10
                elif not focus_finished:
                    focusing(max_index)
                    focus_finished = True
                    print(f"Focus locked at position: {max_index}")
            else:
                skip_frame -= 1

            keyCode = cv2.waitKey(16) & 0xff

            if keyCode == 27:  # ESC - quit
                break
            elif keyCode == 10 or keyCode == 13:  # ENTER - refocus
                max_index = 10
                max_value = 0.0
                last_value = 0.0
                dec_count = 0
                focal_distance = 10
                focus_finished = False
                print("Refocusing...")
            elif keyCode == ord('t') or keyCode == ord('T'):  # T - toggle text detection
                text_detection_on = not text_detection_on
                print(f"Text detection: {'ON' if text_detection_on else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()
    else:
        print('Unable to open camera')

def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--i2c-bus', type=int, nargs=None, required=True,
                        help='Set i2c bus, for A02 is 6, for B01 is 7 or 8, '
                             'for Jetson Xavier NX it is 9 and 10.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_cmdline()
    focuser = Focuser(args.i2c_bus)
    show_camera()
