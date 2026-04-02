import argparse
from Autofocus import show_camera
from Focuser import Focuser

def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--i2c-bus', type=int, default=2,
                        help='Set i2c bus (A02=6, B01=7/8, Xavier NX=9/10)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_cmdline()
    import Autofocus
    Autofocus.focuser = Focuser(args.i2c_bus)
    show_camera()
