import cv2
import tempfile

from Focuser import Focuser

focuser = None


def focusing(val):
    focuser.set(Focuser.OPT_FOCUS, val)


def laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def overlay_preview_status(img, focus_finished):
    preview = img.copy()
    status = 'Focus Locked' if focus_finished else 'Autofocusing'
    cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(preview, 'Enter: send image to Ollama  R: refocus  Esc: quit', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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
            if on_capture:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    image_path = tmp.name
                cv2.imwrite(image_path, img)
                on_capture(image_path)
        elif key in (ord('r'), ord('R')):
            af = _reset_autofocus()
            focusing(af['focal_distance'])

    cap.release()
    cv2.destroyAllWindows()
