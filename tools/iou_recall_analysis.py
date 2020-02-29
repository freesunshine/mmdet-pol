import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.core import eval_map


def get_iou_recall_curve(cfg_path, pth_path, out_path):
    cfg = mmcv.Config.fromfile(cfg_path)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.

    distributed = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, pth_path, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model.eval()
    results = [None for _ in range(len(dataset))]
    print(dataset)
    for idx in range(0, len(dataset), 1):
        data = dataset[idx]['img'][0]
        data = data.cuda()

        # compute output
        with torch.no_grad():
            result = model(data)
        results[idx] = result

    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]

    mean_ap, eval_results = eval_map(
                                    results,
                                    annotations,
                                    scale_ranges=None,
                                    iou_thr=0.5,
                                    dataset=None,
                                    logger='print',
                                    nproc=4)


if __name__ == '__main__':
    cfg_path = '/home/wangyong/Code/mmdet-pol/configs/PolNet/faster_rcnn_pol_r50_fpn_1x_48-96-32-16-1.py'
    pth_path = '/home/wangyong/DataDisk/work_dirs/car-xmls/car1_faster_rcnn_polnet_r50_fpn_1x_48-96-32-16-1/latest.pth'
    out_path = None
    get_iou_recall_curve(cfg_path, pth_path, out_path)