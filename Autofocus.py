import cv2
import os
import re
import shutil
import subprocess
import tempfile
from Focuser import Focuser

focuser = None
def focusing(val):
	# value = (val << 4) & 0x3ff0
	# data1 = (value >> 8) & 0x3f
	# data2 = value & 0xf0
	# os.system("i2cset -y 6 0x0c %d %d" % (data1,data2))
    focuser.set(Focuser.OPT_FOCUS, val)
	
def sobel(img):
	img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	img_sobel = cv2.Sobel(img_gray,cv2.CV_16U,1,1)
	return cv2.mean(img_sobel)[0]

def laplacian(img):
	img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	img_sobel = cv2.Laplacian(img_gray,cv2.CV_16U)
	return cv2.mean(img_sobel)[0]


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


# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps 
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=640, display_height=360, framerate=60, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

def show_camera():
    max_index = 10
    max_value = 0.0
    last_value = 0.0
    dec_count = 0
    focal_distance = 10
    focus_finished = False
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    focusing(focal_distance)
    skip_frame = 2
    if cap.isOpened():
        cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty('CSI Camera',0) >= 0:
            ret_val, img = cap.read()
            if not ret_val:
                continue

            img = cv2.flip(img, -1)
            cv2.imshow('CSI Camera', overlay_preview_status(img, focus_finished))
            
            if skip_frame == 0:
                skip_frame = 2
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
            else:
                skip_frame = skip_frame - 1

            keyCode = cv2.waitKey(1) & 0xff
            if keyCode == 27:
                break
            elif keyCode in (10, 13):
                annotated, processed, text_found, extracted_text = process_capture(img)
                cv2.imshow('Processed Text', annotated)
                cv2.imshow('Processed Mask', processed)
                if extracted_text:
                    print('\nCaptured text:\n{}\n'.format(extracted_text))
                elif text_found:
                    print('\nText-like regions found, but no OCR engine is available. Install pytesseract or tesseract.\n')
                else:
                    print('\nNo text detected in the captured frame.\n')
            elif keyCode in (ord('r'), ord('R')):
                max_index = 10
                max_value = 0.0
                last_value = 0.0
                dec_count = 0
                focal_distance = 10
                focus_finished = False
                focusing(focal_distance)
        cap.release()
        cv2.destroyAllWindows()
    else:
        print('Unable to open camera')
