import cv2
import numpy as np
from matplotlib import pyplot as plt
from prelib import preprocess
import pywt


def segment(img):
    seg = []
    for i in range(4):
        for j in range(4):
            start = i*128
            start1 = j*128
            end = start+128
            end1 = start1+128
            seg.append(img[start:end,start1:end1])
    return seg

def localfeature(seg):
    ll2=[]
    lh1=[]
    hl1=[]
    hh1=[]
    dwvt=[]
    ht = []
    vt = []

    for i in range(16):
        wp = pywt.WaveletPacket2D(data=seg[i],wavelet='haar',mode='sym')
        lh1.append(wp['v'].data)
        hl1.append(wp['h'].data)
        hh1.append(wp['d'].data)
        level1 = np.hstack((np.vstack((wp['aa'].data,wp['vv'].data)),np.vstack((wp['hh'].data,wp['dd'].data))))
        level2 = np.hstack((np.vstack((wp['aaa'].data,wp['vvv'].data)),np.vstack((wp['hhh'].data,wp['ddd'].data))))
        level3 = np.hstack((np.vstack((wp['aaaa'].data,wp['vvvv'].data)),np.vstack((wp['hhhh'].data,wp['dddd'].data))))
        level4 = np.hstack((np.vstack((wp['aaaaa'].data,wp['vvvvv'].data)),np.vstack((wp['hhhhh'].data,wp['ddddd'].data))))
        level3[:8,:8] = level4
        level2[:16,:16] = level3
        level1[:32,:32] = level2
        ll2.append(level1)
        vt.append(np.vstack((ll2[i],lh1[i])))
        ht.append(np.vstack((hl1[i],hh1[i])))
        dwvt.append(np.hstack((vt[i],ht[i])))
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    subvector = []
    vector = []
    for i in range(16):
        s1.append(np.linalg.svd(ll2[i], compute_uv=False))
        s2.append(np.linalg.svd(hl1[i], compute_uv=False))
        s3.append(np.linalg.svd(lh1[i], compute_uv=False))
        s4.append(np.linalg.svd(hh1[i], compute_uv=False))
        subvector.append(np.vstack((np.vstack((s1[i],s2[i])),np.vstack((s3[i],s4[i])))))
        vector.append(subvector[i])
        vector1 = np.concatenate(vector,axis=0)
        vector1 = np.array(vector1,dtype=int)
    return vector1

def globalfeature(img,gran):
    gloseg = np.zeros((gran,gran),dtype=int)
    displ = 512/gran
    for i in range(gran):
        for j in range(gran):
            start = i*displ
            start1 = j*displ
            end = start+displ
            end1 = start1+displ
            if(img[start:end,start1:end1].any()):
                gloseg[i,j]= 1
    return gloseg
