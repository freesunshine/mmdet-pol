import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile
from mmcv.parallel import collate, scatter
from mmcv.runner import Hook
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info, load_checkpoint
from mmdet.apis import init_detector, inference_detector, show_result
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
    model = init_detector(cfg_path, pth_path, device='cuda:0')

    results = [None for _ in range(len(dataset))]
    print(dataset)
    for idx in range(0, len(dataset), 1):
        data = dataset[idx]['img'][0].numpy()
        data = data.transpose((1, 2, 0))

        # compute output
        result = inference_detector(model, data)
        results[idx] = result

    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    out_list = []
    for thr in range(0, 21):
        mean_ap, eval_results = eval_map(
                                        results,
                                        annotations,
                                        scale_ranges=None,
                                        iou_thr=thr/20,
                                        dataset=None,
                                        logger=None,
                                        nproc=1)

        out_list.append([thr/20, eval_results[0]['ap'][-1], eval_results[1]['ap'][-1],mean_ap])
    print(out_list)
    f = open(out_path, 'w')
    for ln in out_list:
        f.write(str(ln[0]))
        f.write(',')
        f.write(str(ln[1]))
        f.write(',')
        f.write(str(ln[2]))
        f.write('\n')
    f.close()


if __name__ == '__main__':
    cfg_path = '/home/wangyong/Code/mmdet-pol/configs/PolNet/faster_rcnn_pol_r50_fpn_1x_48-96-32-16-9.py'
    pth_path = '/home/wangyong/Code/mmdet-pol/work_dirs/person_car2_faster_rcnn_polnet_r50_fpn_1x_48-96-32-16-9/epoch_100.pth'
    out_path = '/home/wangyong/Code/mmdet-pol/work_dirs/iou-courves/48-96-32-16-9.csv'
    get_iou_recall_curve(cfg_path, pth_path, out_path)
