from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import imageio
import os
from timeit import default_timer as timer
import numpy as np


config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
img_dir = ''
output_dir = ''


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

def get_file_paths (folder=None, file_ext=None):
    """ Get the absolute path of all files in given folder recursively
    :param folder:
    :param file_ext:
    :return:
    """
    file_list = []
    if folder is None:
        return file_list
    for file_name in os.listdir(folder):
        if file_name.endswith(file_ext):
            file_list.append(os.path.join(folder, file_name))
    return file_list

def inference(config_file, checkpoint_file, img_paths,  dest_dir, device,
              h_len, w_len, h_overlap, w_overlap, save_res=False):
    model = init_detector(config_file, checkpoint_file, device=device)
    for count, img_path in enumerate(img_paths):

        start = timer()
        img = imageio.imread(img_path)
        img_max_value = np.max(img)  # 查找最大值，用于规范化
        box_res = []
        label_res = []
        score_res = []

        imgH = img.shape[0]
        imgW = img.shape[1]

        if imgH < h_len:
            temp = np.zeros([h_len, imgW, 3], np.float32)
            temp[0:imgH, :, :] = img
            img = temp
            imgH = h_len

        if imgW < w_len:
            temp = np.zeros([imgH, w_len, 3], np.float32)
            temp[:, 0:imgW, :] = img
            img = temp
            imgW = w_len

        for hh in range(0, imgH, h_len - h_overlap):
            if imgH - hh - 1 < h_len:
                hh_ = imgH - h_len
            else:
                hh_ = hh
            for ww in range(0, imgW, w_len - w_overlap):
                if imgW - ww - 1 < w_len:
                    ww_ = imgW - w_len
                else:
                    ww_ = ww
                src_img = img[hh_:(hh_ + h_len), ww_:(ww_ + w_len), :]
                if img_max_value > 255:  # 如果像素值大于255，则规范化
                    src_img = (src_img * (255.0 / img_max_value)).astype('B')
                if len(src_img.shape) == 1:  # 如果是单通道图像，则转为三通道
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
                det_boxes_h_, det_scores_h_, det_category_h_, \
                det_boxes_r_, det_scores_r_, det_category_r_ = \
                    sess.run(
                        [det_boxes_h, det_scores_h, det_category_h,
                         det_boxes_r, det_scores_r, det_category_r],
                        feed_dict={img_plac: src_img[:, :, ::-1]}
                    )

                if len(det_boxes_h_) > 0:
                    for ii in range(len(det_boxes_h_)):
                        box = det_boxes_h_[ii]
                        box[0] = box[0] + ww_
                        box[1] = box[1] + hh_
                        box[2] = box[2] + ww_
                        box[3] = box[3] + hh_
                        box_res.append(box)
                        label_res.append(det_category_h_[ii])
                        score_res.append(det_scores_h_[ii])
                if len(det_boxes_r_) > 0:
                    for ii in range(len(det_boxes_r_)):
                        box_rotate = det_boxes_r_[ii]
                        box_rotate[0] = box_rotate[0] + ww_
                        box_rotate[1] = box_rotate[1] + hh_
                        box_res_rotate.append(box_rotate)
                        label_res_rotate.append(det_category_r_[ii])
                        score_res_rotate.append(det_scores_r_[ii])

        box_res_rotate = np.array(box_res_rotate)
        label_res_rotate = np.array(label_res_rotate)
        score_res_rotate = np.array(score_res_rotate)

        box_res_rotate_ = []
        label_res_rotate_ = []
        score_res_rotate_ = []
        threshold = {'roundabout': 0.1, 'tennis-court': 0.3, 'swimming-pool': 0.1, 'storage-tank': 0.2,
                     'soccer-ball-field': 0.3, 'small-vehicle': 0.2, 'ship': 0.05, 'plane': 0.3,
                     'large-vehicle': 0.1, 'helicopter': 0.2, 'harbor': 0.0001, 'ground-track-field': 0.3,
                     'bridge': 0.0001, 'basketball-court': 0.3, 'baseball-diamond': 0.3}

        for sub_class in range(1, cfgs.CLASS_NUM + 1):
            index = np.where(label_res_rotate == sub_class)[0]
            if len(index) == 0:
                continue
            tmp_boxes_r = box_res_rotate[index]
            tmp_label_r = label_res_rotate[index]
            tmp_score_r = score_res_rotate[index]

            tmp_boxes_r = np.array(tmp_boxes_r)
            tmp = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
            tmp[:, 0:-1] = tmp_boxes_r
            tmp[:, -1] = np.array(tmp_score_r)

            try:
                inx = nms_rotate.nms_rotate_cpu(boxes=np.array(tmp_boxes_r),
                                                scores=np.array(tmp_score_r),
                                                iou_threshold=threshold[LABEL_NAME_MAP[sub_class]],
                                                max_output_size=500)
            except:
                # Note: the IoU of two same rectangles is 0, which is calculated by rotate_gpu_nms
                jitter = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
                jitter[:, 0] += np.random.rand(tmp_boxes_r.shape[0], ) / 1000
                inx = rotate_gpu_nms(np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                     float(threshold[LABEL_NAME_MAP[sub_class]]), 0)

            box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
            score_res_rotate_.extend(np.array(tmp_score_r)[inx])
            label_res_rotate_.extend(np.array(tmp_label_r)[inx])

        time_elapsed = timer() - start

        # 到此，检测完一副图，结果在box_res_rotate_、label_res_rotate_、score_res_rotate中
        # 写入txt
        rboxes = coordinate_convert.forward_convert(box_res_rotate_, with_label=False)
        file_shortname = img_path.split('/')[-1].split('.')[0]
        txt_path = des_folder + '/' + file_shortname + '.txt'
        f_txt = open(txt_path, 'w')
        for i in range(0, len(label_res_rotate_)):
            if score_res_rotate_[i] < cfgs.SHOW_SCORE_THRSHOLD:  # 如果低于阈值，则不输出
                continue
            class_id = int(label_res_rotate_[i])
            class_str = LABEL_NAME_MAP[class_id]
            f_txt.write(class_str)
            f_txt.write(' ')
            for xi in range(0, 8):
                f_txt.write(str(rboxes[i][xi]))
                f_txt.write(' ')
            f_txt.write(str(score_res_rotate_[i]))
            f_txt.write('\n')

        # 输出方框图
        if save_res:
            # det_detections_h = draw_box_in_img.draw_box_cv(np.array(img, np.float32) - np.array(cfgs.PIXEL_MEAN),
            #                                                boxes=np.array(box_res),
            #                                                labels=np.array(label_res),
            #                                                scores=np.array(score_res))
            det_detections_r = draw_box_in_img.draw_rotate_box_cv(np.array(img, np.float32) - np.array(cfgs.PIXEL_MEAN),
                                                                  boxes=np.array(box_res_rotate_),
                                                                  labels=np.array(label_res_rotate_),
                                                                  scores=np.array(score_res_rotate_))
            save_dir = os.path.join(des_folder, 'jpegs')
            tools.mkdir(save_dir)
            # cv2.imwrite(save_dir + '/' + img_path.split('/')[-1].split('.')[0] + '_h.jpg',
            #             det_detections_h)
            cv2.imwrite(save_dir + '/' + img_path.split('/')[-1].split('.')[0] + '_r.jpg',
                        det_detections_r)

            view_bar('{} cost {}s'.format(img_path.split('/')[-1].split('.')[0],
                                          time_elapsed), count + 1, len(file_paths))

        # else:
        #     # eval txt
        #     CLASS_DOTA = NAME_LABEL_MAP.keys()
        #     write_handle = {}
        #     txt_dir = os.path.join('txt_output', cfgs.VERSION)
        #     tools.mkdir(txt_dir)
        #     for sub_class in CLASS_DOTA:
        #         if sub_class == 'back_ground':
        #             continue
        #         write_handle[sub_class] = open(os.path.join(txt_dir, 'Task1_%s.txt' % sub_class), 'a+')
        #
        #     rboxes = coordinate_convert.forward_convert(box_res_rotate_, with_label=False)
        #
        #     for i, rbox in enumerate(rboxes):
        #         command = '%s %.3f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (img_path.split('/')[-1].split('.')[0],
        #                                                                          score_res_rotate_[i],
        #                                                                          rbox[0], rbox[1], rbox[2], rbox[3],
        #                                                                          rbox[4], rbox[5], rbox[6], rbox[7],)
        #         write_handle[LABEL_NAME_MAP[label_res_rotate_[i]]].write(command)
        #
        #     for sub_class in CLASS_DOTA:
        #         if sub_class == 'back_ground':
        #             continue
        #         write_handle[sub_class].close()

        view_bar('{} cost {}s'.format(img_path.split('/')[-1].split('.')[0],
                                      time_elapsed), count + 1, len(file_paths))
        fw.write('{}\n'.format(img_path))
        fw.close()

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
show_result(img, result, model.CLASSES)
# or save the visualization results to image files
show_result(img, result, model.CLASSES, out_file='result.jpg')