import os
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