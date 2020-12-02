
from ctypes import *
import sys
import numpy as np
import os


outw = 768
outh = 448


try:
    pyc_libtest = cdll.LoadLibrary("/usr/lib/libtest_uhwapi.so")
except Exception:
    print('Load libtest_uhwapi.so failed\nPlease set so first!')

pyc_libtest.hardware_init.argtypes = (POINTER(c_void_p), c_char_p, c_int, c_int)

pyc_libtest.preprocess.argtypes = (POINTER(c_void_p), np.ctypeslib.ndpointer(c_uint8), np.ctypeslib.ndpointer(c_int8), c_int, c_int)
pyc_libtest.cust_decode.argtypes = (POINTER(c_void_p), np.ctypeslib.ndpointer(c_int8), np.ctypeslib.ndpointer(c_float), np.ctypeslib.ndpointer(c_float),np.ctypeslib.ndpointer(c_float), c_int, c_float)
pyc_libtest.cust_sfx.argtypes = (POINTER(c_void_p), np.ctypeslib.ndpointer(c_int8), np.ctypeslib.ndpointer(c_float), c_int, c_float)

uhw_handle = c_void_p()


def initial_pp():
    xclbin = "./dpu.xclbin"

    ret = pyc_libtest.hardware_init(uhw_handle, c_char_p(xclbin.encode("utf-8")), outh, outw)

    if ret == -1:
        print("[XPLUSML]  Unable to Create handle for the hardware_init ")
        sys.exit()

def preprocess_input(img, inh, inw):

    imgo = np.empty((outh * outw * 3), dtype=np.dtype(np.int8, align=True))
    pyc_libtest.preprocess(uhw_handle, img, imgo, inh, inw)

    return imgo

def decode_deal(inputdata, xyref, whref, dlength, scale):

    output = np.empty((dlength * 6 * 4), dtype=np.dtype(np.float32, align=True))
    pyc_libtest.cust_decode(uhw_handle, inputdata, output, xyref, whref, dlength, scale)

    return output

def sfx_input(img, dlength, scale):

    imgo = np.empty((dlength * 10), dtype=np.dtype(np.float32, align=True))
    pyc_libtest.cust_sfx(uhw_handle, img, imgo, dlength, scale)

    return imgo
