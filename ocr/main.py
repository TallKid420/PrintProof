import argparse
from Autofocus import show_camera
from Focuser import Focuser


def parse_cmdline():
    parser = argparse.ArgumentParser(description='PrintProof - OCR-only mode')
    parser.add_argument('-i', '--i2c-bus', type=int, default=2,
                        help='Set i2c bus (A02=6, B01=7/8, Xavier NX=9/10)')
    parser.add_argument('--debug', action='store_true',
                        help='Print raw OCR text and focus write operations')
    return parser.parse_args()


def handle_capture(text_found, extracted_text, debug=False):
    if extracted_text:
        print('\nOCR text:\n{}\n'.format(extracted_text))
        return

    if text_found:
        print('\nText regions found but no OCR engine available. Install pytesseract or tesseract.\n')
    else:
        print('\nNo text detected in the captured frame.\n')


if __name__ == '__main__':
    args = parse_cmdline()
    import Autofocus
    Autofocus.focuser = Focuser(args.i2c_bus, debug=args.debug)
    show_camera(on_capture=lambda text_found, extracted_text: handle_capture(text_found, extracted_text, debug=args.debug))
