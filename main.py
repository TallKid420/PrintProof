import argparse
from functools import partial

import Image_llm
from Autofocus import show_camera
from Focuser import Focuser


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--i2c-bus', type=int, default=2,
                        help='Set i2c bus (A02=6, B01=7/8, Xavier NX=9/10)')
    parser.add_argument('--debug', action='store_true',
                        help='Print focus write operations for debugging')
    return parser.parse_args()


def handle_capture(text_found, extracted_text, image_path, debug=False):
    if extracted_text:
        if debug:
            print('\nCaptured text:\n{}\n'.format(extracted_text))
        Image_llm.ProcessImage(image_path)
        return

    if text_found:
        print('\nText-like regions found, but no OCR engine is available. Install pytesseract or tesseract.\n')
    else:
        print('\nNo text detected in the captured frame.\n')


if __name__ == '__main__':
    args = parse_cmdline()
    import Autofocus
    Autofocus.focuser = Focuser(args.i2c_bus, debug=args.debug)
    show_camera(on_capture=partial(handle_capture, debug=args.debug))
