import cv2
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
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


def detect_text(img):
    """Detect text regions in the image using MSER.
    Returns annotated image and True if text-like regions are found."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(5, 60, 2600)
    regions, _ = mser.detectRegions(gray)

    text_found = False
    annotated = img.copy()
    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        aspect = w / float(h) if h > 0 else 0
        # Filter to plausible text character proportions
        if 0.1 < aspect < 10 and w > 8 and h > 8:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 1)
            text_found = True

    label = "Text Detected" if text_found else "No Text"
    color = (0, 255, 0) if text_found else (0, 0, 255)
    cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return annotated, text_found


def extract_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    tesseract_path = shutil.which('tesseract')
    if not tesseract_path:
        return []

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete = False) as temp_file:
            temp_path = temp_file.name

        cv2.imwrite(temp_path, processed)
        result = subprocess.run(
            [tesseract_path, temp_path, 'stdout', '--psm', '6'],
            capture_output = True,
            text = True,
            check = False,
        )
        if result.returncode != 0:
            return []

        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def capture_frame_and_print_text(img):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_path = 'capture_{}.png'.format(timestamp)
    cv2.imwrite(image_path, img)
    print('Saved image: {}'.format(image_path))

    lines = extract_text(img)
    if lines:
        print('Text found:')
        for line in lines:
            print(line)
    else:
        print('No text found.')


def reset_autofocus_state():
    return {
        'max_index': 10,
        'max_value': 0.0,
        'last_value': 0.0,
        'dec_count': 0,
        'focal_distance': 10,
        'focus_finished': False,
        'skip_frame': 4,
        'blur_count': 0,
    }


# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 30fps 
# Flip the image by setting the flip_method (2 rotates 180 degrees)
# display_width and display_height determine the size of the window on the screen

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=30, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

def show_camera():
    state = reset_autofocus_state()
    last_frame = None
    refocus_threshold = 0.75
    blur_limit = 4
    # flip_method=2 rotates the image 180 degrees, which flips both axes.
    print(gstreamer_pipeline(flip_method=2))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    focusing(state['focal_distance'])
    if cap.isOpened():
        cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)
        # Window 
        while cv2.getWindowProperty('CSI Camera',0) >= 0:
            ret_val, img = cap.read()
            if not ret_val:
                continue

            last_frame = img.copy()
            display_img, text_found = detect_text(img)
            cv2.imshow('CSI Camera', display_img)
            
            current_sharpness = laplacian(img)

            if state['skip_frame'] == 0:
                state['skip_frame'] = 4
                if state['dec_count'] < 6 and state['focal_distance'] <= 1000:
                    #Adjust focus
                    focusing(state['focal_distance'])
                    #Find the maximum image clarity
                    if current_sharpness > state['max_value']:
                        state['max_index'] = state['focal_distance']
                        state['max_value'] = current_sharpness
                        
                    #If the image clarity starts to decrease
                    if current_sharpness < state['last_value']:
                        state['dec_count'] += 1
                    else:
                        state['dec_count'] = 0
                    #Image clarity is reduced by six consecutive frames
                    if state['dec_count'] < 6:
                        state['last_value'] = current_sharpness
                        #Increase the focal distance
                        state['focal_distance'] += 10

                elif not state['focus_finished']:
                    #Adjust focus to the best
                    focusing(state['max_index'])
                    state['focus_finished'] = True
                    state['blur_count'] = 0
            else:
                state['skip_frame'] = state['skip_frame'] - 1

            if state['focus_finished'] and state['max_value'] > 0:
                if current_sharpness < state['max_value'] * refocus_threshold:
                    state['blur_count'] += 1
                else:
                    state['blur_count'] = 0

                if state['blur_count'] >= blur_limit:
                    current_focus = focuser.get(Focuser.OPT_FOCUS)
                    state = reset_autofocus_state()
                    state['focal_distance'] = max(0, current_focus - 80)
                    focusing(state['focal_distance'])
            # This also acts as 
            keyCode = cv2.waitKey(16) & 0xff
            # Stop the program on the ESC key
            if keyCode == 27:
                break
            elif keyCode == 13 or keyCode == 10:
                if last_frame is not None:
                    capture_frame_and_print_text(last_frame)
            elif keyCode == ord('r'):
                state = reset_autofocus_state()
                focusing(state['focal_distance'])
        cap.release()
        cv2.destroyAllWindows()
    else:
        print('Unable to open camera')
