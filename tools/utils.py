import os
import sys
import shutil
import cv2
import numpy as np
import torch
import torchvision
import math
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import xml.dom.minidom
from xml.etree import ElementTree

# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh /home/wangyong/Code/mmdet-pol/configs/PolNet/faster_rcnn_bgr_r50_fpn_1x.py 4 --validate
# CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh /home/wangyong/Code/mmdet-pol/configs/PolNet/faster_rcnn_pol_r50_fpn_1x.py 4 --validate


def valid_log(in_path, out_path, memo):
    in_file = open(in_path, 'r')
    out_file = open(out_path, 'w')
    out_file.write(memo)
    out_file.write('\n')
    out_file.write('class gts dets recall ap')
    out_file.write('\n')
    for line in in_file.readlines():
        if line.startswith('|'):
            splited = line.split('|')
            r=[]
            for ch in splited:
                if ch!='':
                    r.append(ch.strip())
            if r[0]!='class' and r[0]!='mAP':
                for c in r:
                    out_file.write(c)
                    out_file.write(' ')
                out_file.write('\n')

valid_log('/home/wangyong/下载/20200129_184158.log', '/home/wangyong/下载/20200129_184158.txt', 'bgr')