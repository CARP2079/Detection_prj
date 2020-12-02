'''
author: wuliyu
data: 2020.12.2
'''

from ctypes import *
import cv2
import numpy as np

import os
import threading
import time

import np_methodsV2
import visualizationV2
import np_anchorV2
import queue
from dnndk import n2cube

import HW_unitedapi


runtimes = 1000


def runpredeal(oimgQueue):


    img_path = '/home/root/testpic/'
    image_names = sorted(os.listdir(img_path))
    runTotal = len(image_names)

    print("start runpredeal thread")

    tpre=0
    tall1=time.time()
    for j in range(runtimes):
        tre1 = time.time()

        orgimg = cv2.imread(img_path + image_names[j])

        tre2 = time.time()
        tpre = tpre+(tre2 - tre1)

        oimgQueue.put(orgimg)

    tall2 = time.time()
    print("all pre time out is cost", (tall2-tall1))
    print("all pre time is cost",tpre)

    return

def hardware_propress(oimgQueue,preedQueue,threadLock):
    print("start hardware_propress thread")
    tpre = 0
    tall1 = time.time()
    for j in range(runtimes):

        tre1 = time.time()
        orgimg=oimgQueue.get()

        threadLock.acquire()
        imgo = HW_unitedapi.preprocess_input(orgimg, orgimg.shape[0], orgimg.shape[1])
        threadLock.release()
        tre2 = time.time()
        tpre = tpre + (tre2 - tre1)

        listsss = []
        listsss.append(orgimg)
        listsss.append(imgo)
        preedQueue.put(listsss)

    tall2 = time.time()
    print("all hardware_propress time out is cost", (tall2 - tall1))
    print("all hardware_propress time is cost",tpre)



"""
                        Input Node(s)   (H*W*C)
ssd_mobilenet_v2_conv2d_conv2d_conv2d_Conv2D(0) : 448*768*3

                          Output Node(s)   (H*W*C)
ssd_mobilenet_v2_block8_box_conv_cls_2_conv_cls_2_Conv2D(0) : 56*96*60
ssd_mobilenet_v2_block8_box_conv_loc_2_conv_loc_2_Conv2D(0) : 56*96*24
ssd_mobilenet_v2_block7_box_conv_cls_2_conv_cls_2_Conv2D(0) : 28*48*60
ssd_mobilenet_v2_block7_box_conv_loc_2_conv_loc_2_Conv2D(0) : 28*48*24
ssd_mobilenet_v2_block6_box_conv_cls_2_conv_cls_2_Conv2D(0) : 14*24*60
ssd_mobilenet_v2_block6_box_conv_loc_2_conv_loc_2_Conv2D(0) : 14*24*24
ssd_mobilenet_v2_block5_box_conv_cls_2_conv_cls_2_Conv2D(0) : 7*12*60
ssd_mobilenet_v2_block5_box_conv_loc_2_conv_loc_2_Conv2D(0) : 7*12*24


"""

def runDPU(preedQueue, dpuresQueue, threadnum):


    KERNEL_CONV="testnet"
    CONV_INPUT_NODE="ssd_mobilenet_v2_conv2d_conv2d_conv2d_Conv2D"

    CONV_OUTPUT_NODE=[]
    CONV_OUTPUT_NODE.append("ssd_mobilenet_v2_block8_box_conv_cls_2_conv_cls_2_Conv2D")
    CONV_OUTPUT_NODE.append("ssd_mobilenet_v2_block7_box_conv_cls_2_conv_cls_2_Conv2D")
    CONV_OUTPUT_NODE.append("ssd_mobilenet_v2_block6_box_conv_cls_2_conv_cls_2_Conv2D")
    CONV_OUTPUT_NODE.append("ssd_mobilenet_v2_block5_box_conv_cls_2_conv_cls_2_Conv2D")
    CONV_OUTPUT_NODE.append("ssd_mobilenet_v2_block8_box_conv_loc_2_conv_loc_2_Conv2D")
    CONV_OUTPUT_NODE.append("ssd_mobilenet_v2_block7_box_conv_loc_2_conv_loc_2_Conv2D")
    CONV_OUTPUT_NODE.append("ssd_mobilenet_v2_block6_box_conv_loc_2_conv_loc_2_Conv2D")
    CONV_OUTPUT_NODE.append("ssd_mobilenet_v2_block5_box_conv_loc_2_conv_loc_2_Conv2D")


    reshapsize=[]
    reshapsize.append((1, 56, 96, 6, 10))
    reshapsize.append((1, 28, 48, 6, 10))
    reshapsize.append((1, 14, 24, 6, 10))
    reshapsize.append((1, 7, 12, 6, 10))
    reshapsize.append((1, 56, 96, 6, 4))
    reshapsize.append((1, 28, 48, 6, 4))
    reshapsize.append((1, 14, 24, 6, 4))
    reshapsize.append((1, 7, 12, 6, 4))


    """ Attach to DPU driver and prepare for running """
    n2cube.dpuOpen()

    """ Create DPU Kernels for CONV NODE in imniResNet """
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)

    """ Create DPU Tasks for CONV NODE in miniResNet """
    task = n2cube.dpuCreateTask(kernel, 0)

    conv_sbbox_size=[]
    for i in range(8):
        conv_sbbox_size.append( n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE[i]))
        print("outputdata %d is %d" % (i, conv_sbbox_size[i]))

    print("finish set dpu")

    #tstart=time.time()
    tdpu=0
    tall1 = time.time()
    for j in range(runtimes):


        listin = preedQueue.get()
        orgimg=listin[0]
        imgo=listin[1]

        tpost1 = time.time()
        # print("preedimg.shape is",preedimg.shape)


        n2cube.dpuSetInputTensorInHWCInt8(task, CONV_INPUT_NODE, imgo, 1032192)  #448*768*3

        """  Launch miniRetNet task """
        # print("ready to start dpu work")
        n2cube.dpuRunTask(task)

        outputData = []
        outputData.append(orgimg)
        for i in range(8):
            conv_out = n2cube.dpuGetOutputTensorInHWCInt8(task, CONV_OUTPUT_NODE[i], conv_sbbox_size[i])
            conv_out = np.reshape(conv_out, reshapsize[i])
            outputData.append(conv_out)

        tpost2 = time.time()
        tdpu=tdpu+(tpost2 - tpost1)
        #print("one dpu cost time is", (tpost2 - tpost1))
        dpuresQueue.put(outputData)
    tall2 = time.time()
    print("all dpu time out is cost", (tall2 - tall1))
    print("all dpu cost time is", tdpu)

    n2cube.dpuDestroyTask(task)

    return

def hardware_posrpress(dpuresQueue,postedQueue,threadLock):
    net_shape = (448, 768)
    ssd_anchors_h = np_anchorV2.anchors(net_shape)

    outputScale = []
    outputScale.append(0.125000)
    outputScale.append(0.125000)
    outputScale.append(0.125000)
    outputScale.append(0.250000)

    dcoutputscale = []
    dcoutputscale.append(0.062500)
    dcoutputscale.append(0.062500)
    dcoutputscale.append(0.062500)
    dcoutputscale.append(0.062500)

    reshaplen = []
    reshaplen.append(56 * 96 * 6)
    reshaplen.append(28 * 48 * 6)
    reshaplen.append(14 * 24 * 6)
    reshaplen.append(7 * 12 * 6)

    dlength = []
    dlength.append(56 * 96)
    dlength.append(28 * 48)
    dlength.append(14 * 24)
    dlength.append(7 * 12)

    reshapsize = []
    reshapsize.append((1, 56, 96, 6, 4))
    reshapsize.append((1, 28, 48, 6, 4))
    reshapsize.append((1, 14, 24, 6, 4))
    reshapsize.append((1, 7, 12, 6, 4))

    sfxreshapsize = []
    sfxreshapsize.append((1, 56, 96, 6, 10))
    sfxreshapsize.append((1, 28, 48, 6, 10))
    sfxreshapsize.append((1, 14, 24, 6, 10))
    sfxreshapsize.append((1, 7, 12, 6, 10))

    print("start  hardware_posrpress thread")
    thwpost=0
    tall1 = time.time()
    for j in range(runtimes):
        outputData = dpuresQueue.get()
        # print("show it get outdata")
        tpost1 = time.time()

        orimg = outputData[0]
        rlogits = outputData[1:5]
        rlocalisations_int8 = outputData[5:]
        postresdata=[]
        postresdata.append(orimg)

        ## softmax
        threadLock.acquire()
        for j in range(4):
            outarraysfx = HW_unitedapi.sfx_input(rlogits[j], reshaplen[j], outputScale[j])
            outarraysfx = np.reshape(outarraysfx, sfxreshapsize[j])
            postresdata.append(outarraysfx)

        for j in range(4):
            outarraydc = HW_unitedapi.decode_deal(rlocalisations_int8[j], ssd_anchors_h[j][0], ssd_anchors_h[j][1], dlength[j],
                                           dcoutputscale[j])
            outarraydc = np.reshape(outarraydc, reshapsize[j])
            postresdata.append(outarraydc)
        threadLock.release()

        tpost2 = time.time()
        thwpost = thwpost + (tpost2 - tpost1)
        postedQueue.put(postresdata)

    tall2 = time.time()
    print("all hardware_posrpress time out is cost", (tall2 - tall1))
    print("hardware_posrpress cost time is", thwpost)



def software_posrpress(postedQueue,swpostedQueue):
    select_threshold = 0.5
    nms_threshold = 0.4

    rbbox_img = np.array([0., 0., 1., 1.])

    tpost = 0
    tall1 = time.time()
    for j in range(runtimes):
        # print("show it get orimg pic")
        outputData = postedQueue.get()
        # print("show it get outdata")
        tpost1 = time.time()

        orimg = outputData[0]
        rpredictions = outputData[1:5]
        rlocalisations = outputData[5:]

        # step4  bbox decode 解码，因为原bbox的值是相对于 anchor的偏移量， 现在要解码成 xmin xmax ymin ymax 数值
        rclasses, rscores, rbboxes = np_methodsV2.ssd_bboxes_select(
            rpredictions, rlocalisations, select_threshold=select_threshold)

        # step5  # 检测有没有超出检测边缘
        rbboxes = np_methodsV2.bboxes_clip(rbbox_img, rbboxes)
        # step6
        rclasses, rscores, rbboxes = np_methodsV2.bboxes_sort(rclasses, rscores, rbboxes, top_k=300)
        # 去重，将重复检测到的目标去掉
        # step7
        rclasses, rscores, rbboxes = np_methodsV2.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)

        orimg=visualizationV2.plt_bboxes123(orimg, rclasses, rscores, rbboxes)

        tpost2 = time.time()
        tpost = tpost + (tpost2 - tpost1)

        swpostedQueue.put(orimg)
    tall2 = time.time()
    print("all software_posrpress out is cost", (tall2 - tall1))
    print("all software_posrpress is", tpost)

    return


def runshowit(swpostedQueue):

    tpost = 0
    tall1 = time.time()
    for j in range(runtimes):
        imgo = swpostedQueue.get()

        tpost1 = time.time()

        cv2.imshow("frame", imgo)
        cv2.waitKey(10)

        tpost2 = time.time()
        tpost = tpost + (tpost2 - tpost1)

    tall2 = time.time()
    print("all show time out is cost", (tall2 - tall1))
    print("all show is", tpost)

    return

def main():

    oimgQueue = queue.Queue(3)
    preedQueue = queue.Queue(3)
    postedQueue = queue.Queue(3)
    dpuresQueue = queue.Queue(3)
    swpostedQueue = queue.Queue(3)
    threadLock = threading.Lock()

    HW_unitedapi.initial_pp()

    t1 = threading.Thread(target=runpredeal, args=(oimgQueue,))

    tt1 = threading.Thread(target=hardware_propress, args=(oimgQueue, preedQueue,threadLock))

    t2 = threading.Thread(target=runDPU, args=(preedQueue, dpuresQueue,  1))

    tt2 = threading.Thread(target=hardware_posrpress, args=(dpuresQueue, postedQueue,threadLock))

    t3 = threading.Thread(target=software_posrpress, args=(postedQueue, swpostedQueue))

    t4 = threading.Thread(target=runshowit, args=(swpostedQueue,))

    tallstart = time.time()
    tt1.start()
    tt2.start()

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t4.join()
    t1.join()
    t2.join()
    t3.join()
    tt1.join()
    tt2.join()
    tallfinish=time.time()
    print("main all cost time is",(tallfinish-tallstart))
    return


if __name__ == '__main__':
    main()