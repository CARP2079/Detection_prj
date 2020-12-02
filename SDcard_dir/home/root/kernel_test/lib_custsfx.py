'''
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''


from ctypes import *
import sys
import numpy as np
import cv2
import os
import time
import HW_unitedapi



def main():

    HW_unitedapi.initial_pp()

    print("sfx main start!")
    outputScale = []
    outputScale.append(0.125000)
    outputScale.append(0.125000)
    outputScale.append(0.125000)
    outputScale.append(0.250000)

    reshapsize = []
    reshapsize.append((1, 56, 96, 6, 10))
    reshapsize.append((1, 28, 48, 6, 10))
    reshapsize.append((1, 14, 24, 6, 10))
    reshapsize.append((1, 7, 12, 6, 10))

    reshaplen = []
    reshaplen.append(56*96*6)
    reshaplen.append(28*48*6)
    reshaplen.append(14*24*6)
    reshaplen.append(7*12*6)

    rlogits = []
    rpredictions = []


    for i in range(4):
        name = './mysfxdata/rlogits' + str(i) + '.npy'
        bb = np.load(name)
        rlogits.append(bb)

    timea1 = time.time()
    for j in range(4):
        outarray = HW_unitedapi.sfx_input(rlogits[j], reshaplen[j], outputScale[j])
        rpredictions.append(outarray)
    timea2 = time.time()

    print("all 4 softmax cost is",(timea2-timea1))

    for m in range(4):
        name = './mysfxdata/rpredictions' + str(m) + '.npy'
        gold = np.load(name)
        golddata=gold.ravel()
        
        mydata=rpredictions[m]
        
        print("outdifference max that is ", (np.amax((golddata - mydata).ravel(), 0)))


    #print("max that is ",(np.amax((gold.ravel()-outarray).ravel(),0)))


if __name__ == '__main__':
    main()

