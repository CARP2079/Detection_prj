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

outsize = 224
outw=768
outh=448

#outw=224
#outh=224





def main():
    print("py main start!")

    HW_unitedapi.initial_pp()

    img_path = '../testcolor/'
    _R_MEAN = 123.
    
    _G_MEAN = 117.
    _B_MEAN = 104.
    
    EVAL_SIZE = (outw, outh)

    image_names = sorted(os.listdir(img_path))
    orgimg = cv2.imread(img_path + image_names[1])
    timex1=time.time()
    image = cv2.resize(orgimg, EVAL_SIZE, interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preedimgf = np.subtract(image, np.array([_R_MEAN, _G_MEAN, _B_MEAN], np.float32))

    timex2=time.time()
    print("time soft cost is",(timex2-timex1))
    

    #imgo=np.empty((outsize,outsize,3),dtype=np.int16)

    print(orgimg.shape[0],orgimg.shape[1])
    print("orgimg is",orgimg[0,0,0],orgimg[0,0,1],orgimg[0,0,2])

    imgo=HW_unitedapi.preprocess_input(orgimg,orgimg.shape[0],orgimg.shape[1])

    imgo=np.reshape(imgo,(outh,outw,3))

    imgo=imgo.astype(np.float32)
    imgo=imgo*2
    
    print(preedimgf[0,0,0],preedimgf[0,0,1],preedimgf[0,0,2])
    
    cv2.imshow("hello",preedimgf)
    cv2.imshow("hello1",imgo)
    cv2.waitKey(4000)

    print("start!")

    for kk in range(4):
        orgimg = cv2.imread(img_path + image_names[kk])
        imgo=HW_unitedapi.preprocess_input(orgimg,orgimg.shape[0],orgimg.shape[1])
        imgo=imgo.astype(np.float32)
        imgo=imgo*2
        image = cv2.resize(orgimg, EVAL_SIZE, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preedimgf = np.subtract(image, np.array([_R_MEAN, _G_MEAN, _B_MEAN], np.float32))
        imgo=np.reshape(imgo,(outh,outw,3))
        #imgo=imgo/128
        print("this pic is ",kk)
        print("image   o = ",(imgo[0,0,0]),(imgo[0,0,1]),(imgo[0,0,2]))
        print("preedimgf = ",preedimgf[0,0,0],preedimgf[0,0,1],preedimgf[0,0,2])
        cv2.imshow("hello1",imgo)
        cv2.imshow("hello",preedimgf)
        print("finish one")
        
        cv2.waitKey(4000)



if __name__ == '__main__':
    main()

