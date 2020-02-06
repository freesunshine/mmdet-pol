from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import sys
from mmdet.datasets.pipelines.loading import LoadPolNPZImageFromFile
from mmdet.datasets.pipelines.loading import LoadPolSubImageFromFile


def test_and_draw_from_single_file(sample_file, ext_name, bgr_file, out_file, config_file, checkpoint_file, score_threhold):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    sample = None
    if ext_name == 'bgr':
        sample = mmcv.imread(sample_file)
    elif ext_name == 'tiff':
        sample = LoadPolSubImageFromFile(sample_file)
    else:
        sample = LoadPolNPZImageFromFile(sample_file)

    result = inference_detector(model, sample)
    img = mmcv.imread(bgr_file)
    show_result(img, result, model.CLASSES, out_file=out_file, score_thr=score_threhold)


def test_and_draw_from_xmls(xml_dir, ext_name, sample_dir, bgr_dir, out_dir, config_file, checkpoint_file, score_threhold):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    xlms = os.listdir(xml_dir)
    sample_ids = [i.split('_')[0]+'_'+i.split('_')[1] for i in xlms if i.endswith('.xml')]

    for xlm_filename in xlms:
        sample_file=''
        sample_path=''
        sample = None
        if ext_name=='bgr':
            sample_file = xlm_filename.split('.')[0] + '.tiff'
        else:
            sample_file = xlm_filename.split('_')[0] + '_' + xlm_filename.split('_')[1] + '.' + 'ext_name'

        sample_path = os.path.join(sample_dir, sample_file)
        if ext_name=='bgr':
            sample = mmcv.imread(sample_path)
        elif ext_name=='tiff':
            sample = LoadPolSubImageFromFile(sample_path)
        else:
            sample = LoadPolNPZImageFromFile(sample_path)

        img_path = os.path.join(bgr_dir, xlm_filename.split('.')[0] + '.tiff')
        img = mmcv.imread(img_path)

        result = inference_detector(model, sample)

        out_file = xlm_filename.split('_')[0] + '_' + xlm_filename.split('_')[1] + '.' + ext_name+'.jpg'
        out_path = os.path.join(out_dir, out_file)

        show_result(img, result, model.CLASSES, out_file=out_path, score_thr=score_threhold)

        print(out_path)