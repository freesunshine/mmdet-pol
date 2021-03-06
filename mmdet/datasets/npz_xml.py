import os.path as osp
import os
import xml.etree.ElementTree as ET
import numpy as np
from .custom import CustomDataset
from .registry import DATASETS

@DATASETS.register_module
class NPZDataset(CustomDataset):
    CLASSES = ('car',)

    def __init__(self, min_size=None, pic_fmt='.npz', classes=('bicycle', 'car', 'person', 'bus'),  **kwargs):
        self.pic_fmt = pic_fmt
        self.min_size = min_size
        super(NPZDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.CLASSES = classes

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = [i.split('/')[-1].split('.')[0] for i in os.listdir(ann_file) if i.endswith('.xml')]

        for img_id in img_ids:
            tif_name = img_id.split('_')[0] + '_' + img_id.split('_')[1]
            filename = '{}{}'.format(tif_name, self.pic_fmt)
            xml_path = osp.join(self.ann_file, '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()

            is_in_list=False
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name in self.CLASSES:
                    is_in_list = True
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            if is_in_list:
                img_infos.append(
                    dict(id=img_id, filename=filename, width=width, height=height))

        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.ann_file, '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in self.CLASSES:
                label = self.cat2label[name]
                difficult = int(obj.find('difficult').text)
                bnd_box = obj.find('bndbox')
                bbox = [
                    int(bnd_box.find('xmin').text),
                    int(bnd_box.find('ymin').text),
                    int(bnd_box.find('xmax').text),
                    int(bnd_box.find('ymax').text)
                ]
                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True
                if difficult or ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
