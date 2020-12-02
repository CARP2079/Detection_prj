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
import np_anchorV2
import np_anchor
import np_methods
import HW_unitedapi




def main():

    HW_unitedapi.initial_pp()

    print("py main start!")
    net_shape = (448, 768)
    select_threshold = 0.5
    nms_threshold = 0.4

    outputscale=[]
    outputscale.append(0.062500)
    outputscale.append(0.062500)
    outputscale.append(0.062500)
    outputscale.append(0.062500)

    reshapsize = []
    reshapsize.append((1, 56, 96, 6, 4))
    reshapsize.append((1, 28, 48, 6, 4))
    reshapsize.append((1, 14, 24, 6, 4))
    reshapsize.append((1, 7, 12, 6, 4))

    dlength=[]
    dlength.append(56 * 96)
    dlength.append(28 * 48)
    dlength.append(14 * 24)
    dlength.append(7 * 12)

    ssd_anchors_g = np_anchor.anchors(net_shape)
    ssd_anchors_h = np_anchorV2.anchors(net_shape)

    rlocalisations_gold = []
    rlocalisations_gold_res = []
    for i in range(4):
        name = './mysfxdata/rlogits' + str(i + 4) + '.npy'
        bb = np.load(name)
        bb = bb * outputscale[i]
        bb = bb.astype(np.float32)
        bb = np.reshape(bb, reshapsize[i])
        rlocalisations_gold.append(bb)

    timea1 = time.time()
    for i in range(4):
        decodeout = np_methods.ssd_bboxes_decode(rlocalisations_gold[i], ssd_anchors_g[i])
        rlocalisations_gold_res.append(decodeout)
    timea2 = time.time()
    print("sw all 4 softmax cost is",(timea2-timea1))
    

    rlocalisations_int8 = []
    rlocalisations_int8_res = []

    for i in range(4):
        name = './mysfxdata/rlogits' + str(i+4) + '.npy'
        bb = np.load(name)
        rlocalisations_int8.append(bb)



    timea1 = time.time()
    for j in range(4):
        outarray = HW_unitedapi.decode_deal(rlocalisations_int8[j],ssd_anchors_h[j][0],ssd_anchors_h[j][1],dlength[j],outputscale[j])
        outarray=np.reshape(outarray, reshapsize[j])
        rlocalisations_int8_res.append(outarray)

    timea2 = time.time()

    print("all 4 softmax cost is",(timea2-timea1))

    for m in range(4):
        golddata=rlocalisations_gold_res[m].ravel()
        mydata=rlocalisations_int8_res[m].ravel()
        print(golddata.shape,mydata.shape)
        print("outdifference max that is ", (np.amax((golddata - mydata), 0)),(np.amin((golddata - mydata), 0)))
        #idxes = np.where((golddata - mydata) > 0.5)
        #print(idxes)




    #print("max that is ",(np.amax((gold.ravel()-outarray).ravel(),0)))


if __name__ == '__main__':
    main()

