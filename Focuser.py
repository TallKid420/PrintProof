import sys
import time
import os

class Focuser:
    bus = None
    CHIP_I2C_ADDR = 0x0C

    def __init__(self, bus):
        self.focus_value = 0
        self.bus = bus
        pass
        
    def read(self):
        return self.focus_value

    def write(self, chip_addr, value):
        if value < 0:
            value = 0
        self.focus_value = value

        value = (value << 4) & 0x3ff0
        data1 = (value >> 8) & 0x3f
        data2 = value & 0xf0
        os.system("i2cset -y {} 0x{:02X} {} {}".format(self.bus, chip_addr, data1, data2))

    OPT_BASE    = 0x1000
    OPT_FOCUS   = OPT_BASE | 0x01
    OPT_ZOOM    = OPT_BASE | 0x02
    OPT_MOTOR_X = OPT_BASE | 0x03
    OPT_MOTOR_Y = OPT_BASE | 0x04
    OPT_IRCUT   = OPT_BASE | 0x05
    opts = {
        OPT_FOCUS : {
            "MIN_VALUE": 0,
            "MAX_VALUE": 1000,
            "DEF_VALUE": 0,
        },
    }
    def reset(self,opt,flag = 1):
        info = self.opts[opt]
        if info == None or info["DEF_VALUE"] == None:
            return
        self.set(opt,info["DEF_VALUE"])

    def get(self,opt,flag = 0):
        info = self.opts[opt]
        return self.read()

    def set(self,opt,value,flag = 1):
        info = self.opts[opt]
        if value > info["MAX_VALUE"]:
            value = info["MAX_VALUE"]
        elif value < info["MIN_VALUE"]:
            value = info["MIN_VALUE"]
        self.write(self.CHIP_I2C_ADDR, value)
        print("write: {}".format(value))

pass 

def test():
    focuser = Focuser(7)
    focuser.set(Focuser.OPT_FOCUS, 0)
    time.sleep(3)
    focuser.set(Focuser.OPT_FOCUS, 1000)
    time.sleep(3)
    focuser.reset(Focuser.OPT_FOCUS)

if __name__ == "__main__":
    test()
