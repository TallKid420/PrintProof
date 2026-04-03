import cv2
import os
import re
import shutil
import subprocess
import tempfile
from Focuser import Focuser

focuser = None


def focusing(val):
    focuser.set(Focuser.OPT_FOCUS, val)


def sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.mean(cv2.Sobel(gray, cv2.CV_16U, 1, 1))[0]


def laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def preprocess_text_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    threshold = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    inverted = cv2.bitwise_not(threshold)
    repaired = cv2.morphologyEx(
        inverted,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return cv2.bitwise_not(repaired)


def detect_text(img, processed=None):
    if processed is None:
        processed = preprocess_text_image(img)

    grouped = cv2.dilate(
        cv2.bitwise_not(processed),
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
        iterations=1,
    )
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_found = False
    annotated = img.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / float(h) if h > 0 else 0
        if 80 <= area <= 60000 and 0.3 <= aspect <= 20 and h >= 10:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text_found = True

    label = "Text Detected" if text_found else "No Text"
    color = (0, 255, 0) if text_found else (0, 0, 255)
    cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return annotated, text_found


def extract_text(processed):
    try:
        import pytesseract
        return pytesseract.image_to_string(processed, config='--oem 3 --psm 6')
    except ImportError:
        pass

    tesseract_bin = shutil.which('tesseract')
    if tesseract_bin is None:
        return ''

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        cv2.imwrite(temp_path, processed)
        result = subprocess.run(
            [tesseract_bin, temp_path, 'stdout', '--oem', '3', '--psm', '6'],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def format_text(text):
    text = text.replace('\r\n', '\n')
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    paragraphs = []
    for block in re.split(r'\n\s*\n', text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        merged_lines = [lines[0]]
        for line in lines[1:]:
            if re.search(r'[.!?:)]$', merged_lines[-1]):
                merged_lines.append(line)
            else:
                merged_lines[-1] = '{} {}'.format(merged_lines[-1], line)

        paragraphs.append('\n'.join(merged_lines))

    return re.sub(r'[ \t]+', ' ', '\n\n'.join(paragraphs)).strip()


def process_capture(img):
    processed = preprocess_text_image(img)
    annotated, text_found = detect_text(img, processed=processed)
    extracted_text = format_text(extract_text(processed))

    return annotated, processed, text_found, extracted_text


def overlay_preview_status(img, focus_finished):
    preview = img.copy()
    status = 'Focus Locked' if focus_finished else 'Autofocusing'
    cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(preview, 'Enter: capture text  R: refocus  Esc: quit', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return preview


def gstreamer_pipeline(
    capture_width=1280, capture_height=720,
    display_width=640, display_height=360,
    framerate=60, flip_method=0,
):
    return (
        'nvarguscamerasrc ! '
        'video/x-raw(memory:NVMM), '
        'width=(int)%d, height=(int)%d, '
        'format=(string)NV12, framerate=(fraction)%d/1 ! '
        'nvvidconv flip-method=%d ! '
        'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
        'videoconvert ! '
        'video/x-raw, format=(string)BGR ! appsink'
        % (capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

def _reset_autofocus():
    return {
        'max_index': 10,
        'max_value': -1.0,
        'last_value': -1.0,
        'dec_count': 0,
        'focal_distance': 10,
        'focus_finished': False,
        'settle_frames': 3,
    }


def show_camera(on_capture=None):
    af = _reset_autofocus()

    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print('Unable to open camera')
        return

    focusing(af['focal_distance'])
    cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)

    while cv2.getWindowProperty('CSI Camera', 0) >= 0:
        ret_val, img = cap.read()
        if not ret_val:
            continue

        img = cv2.flip(img, -1)
        cv2.imshow('CSI Camera', overlay_preview_status(img, af['focus_finished']))

        if not af['focus_finished']:
            if af['settle_frames'] > 0:
                af['settle_frames'] -= 1
            else:
                val = laplacian(img)
                if val > af['max_value']:
                    af['max_index'] = af['focal_distance']
                    af['max_value'] = val

                if af['last_value'] >= 0 and val < af['last_value']:
                    af['dec_count'] += 1
                else:
                    af['dec_count'] = 0

                af['last_value'] = val

                if af['dec_count'] >= 6 or af['focal_distance'] >= 1000:
                    focusing(af['max_index'])
                    af['focus_finished'] = True
                else:
                    af['focal_distance'] += 10
                    focusing(af['focal_distance'])
                    af['settle_frames'] = 3

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (10, 13):
            annotated, processed, text_found, extracted_text = process_capture(img)
            cv2.imshow('Processed Text', annotated)
            cv2.imshow('Processed Mask', processed)
            if on_capture:
                on_capture(text_found, extracted_text)
        elif key in (ord('r'), ord('R')):
            af = _reset_autofocus()
            focusing(af['focal_distance'])

    cap.release()
    cv2.destroyAllWindows()
