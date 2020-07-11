import os
import numpy as np
from PIL import Image
from sklearn.metrics import normalized_mutual_info_score

pth_dir = "/home/wangyong/Data"
out_dir = "/home/wangyong/data/123/out/"
sample_names = [i for i in os.listdir(pth_dir) if i.endswith('.jpg')]
sample_paths = [os.path.join(pth_dir, i) for i in sample_names]

mi_list={}
for arr1 in sample_names:
    arr1_path = os.path.join(pth_dir, arr1)
    img1 = np.array(Image.open(arr1_path).convert('L'))[406:535, 707:908]
    for arr2 in sample_names:
        arr2_path = os.path.join(pth_dir, arr2)
        img2 = np.array(Image.open(arr2_path).convert('L'))[406:535, 707:908]
        ss=Image.fromarray(img2)
        ss.save("/home/wangyong/Data/ss.jpg")
        mi = normalized_mutual_info_score(img1.ravel(), img2.ravel())
        mi_list[arr1+"-"+arr2] = mi

for ii in mi_list:
    print(ii,":\t",  mi_list[ii])

