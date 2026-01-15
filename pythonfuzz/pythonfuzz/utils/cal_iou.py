import tensorflow as tf
from pathlib import Path
# import sys
import numpy as np
import glob
import os

def cal_iou_xyxy(box1,box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
  
    assert x1min <= 1920 and x1max <= 1920 and y1max<=1080 and y1min<=1080
    #计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    #计算相交部分的坐标
    xmin = max(x1min,x2min)
    ymin = max(y1min,y2min)
    xmax = min(x1max,x2max)
    ymax = min(y1max,y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    #计算iou
    iou = intersection / union
    return iou

# def calc_iou(self, boxes1, boxes2, scope='iou'):
#     """calculate ious
#     Args:
#         boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> 4：(x_center, y_center, w, h)
#         （2,7,7,2,4）
#         boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> 4：(x_center, y_center, w, h)
#         （2,7,7,2,4）
#     Return:
#         iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]  --（2,7,7,2）
#     """
#     with tf.variable_scope(scope):
#         # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
#         boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
#                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
#                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
#                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
#                             axis=-1)  #tf.stack：矩阵拼接

#         boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
#                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
#                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
#                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
#                             axis=-1)

#         # calculate the left up point & right down point
#         lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2]) #左上角坐标最大值
#         rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:]) #右下角坐标最小值

#         # intersection
#         intersection = tf.maximum(0.0, rd - lu)
#         inter_square = intersection[..., 0] * intersection[..., 1]

#         # calculate the boxs1 square and boxs2 square
#         square1 = boxes1[..., 2] * boxes1[..., 3]
#         square2 = boxes2[..., 2] * boxes2[..., 3]

#         union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

#     return tf.clip_by_value(inter_square / union_square, 0.0, 1.0) #截断操作，即如果值不在指定的范围里，那么就会进行最大小值截断




def get_set_iou(predictions_path, all_imgs_ground_truths):
    npy_files = glob.glob(os.path.join(predictions_path, "prediction_*.npy"))
    npy_files.sort()
    imgs_iou = []
    # print('all_imgs_ground_truths', len(all_imgs_ground_truths))
    # all_imgs_ground_truths = np.load(ground_truth_path, allow_pickle=True)  #所有图像的ground_truth
    for i, img_ground_truths in enumerate(all_imgs_ground_truths):    #一个图像的ground——truth，是一个list，里面包含多个红绿灯识别框的信息（一个红绿灯一个框）
        img_ious=[]            #存储一个图像中每个红绿灯的iou值
        results = np.load(npy_files[i], allow_pickle=True).item()    #这个图像对应的模型输出，是一个字典包括labels和boxes
        labels, boxes = results['labels'], results['boxes']
        # print(boxes)
        for ground_truth in img_ground_truths:  #对于每一个识别框, 计算模型输出中的最大iou值
            # print('ground_truth:',ground_truth)
            max_iou = 0
            if 'label' not in ground_truth and 'box' not in ground_truth:
                img_ious.append(1)
            if 'label' in ground_truth:
                for i, label in enumerate(labels):
                    if label == ground_truth['label']:
                        iou = cal_iou_xyxy(boxes[i], ground_truth['box'])
                        max_iou = max(max_iou, iou)
                img_ious.append(max_iou)
            else:
                for box in boxes:
                    iou = cal_iou_xyxy(box, ground_truth['box'])
                    max_iou = max(max_iou, iou)
                img_ious.append(max_iou)
        imgs_iou.append(np.mean(img_ious))
    for i, iou in enumerate(imgs_iou):
        if np.isnan(imgs_iou[i]):
            imgs_iou[i]=1
            
    # iter_iou = np.mean(imgs_iou)
    print('img_ious',imgs_iou)
    return imgs_iou

if __name__ == '__main__':
    # iou = get_set_iou('/media/lzq/D/lzq/pylot_test/pylot/predictions', '/media/lzq/D/lzq/pylot_test/pylot/light_y.npy')
    # print(iou)
    import json
    label = [[{'label': 1, 'box': [1450, 270, 1601, 485]}],[{'label': 1, 'box': [1392, 346, 1498, 500]}]]
    iou = get_set_iou('/media/lzq/D/lzq/pylot_test/pylot/predictions', label)
    print(iou)
    



