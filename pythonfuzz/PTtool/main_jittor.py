import numpy as np
# from fire import Fire
from termcolor import colored
import subprocess
import sys
from pt import TriProCover
from utils.utils import num_to_str
import prettytable
import matplotlib.pyplot as plt


def color_print(s, c):
    print(colored(s, c))

def manual_softmax_row(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def meature(split_num, class_num, data_path, model_path, save_path):
    nb_classes = class_num
    deep_num = split_num
    data_path_pre = "{}/pre.npy".format(data_path)
    data_path_labels = "{}/labels.npy".format(data_path)

    pre = np.load(data_path_pre)
    labels = np.load(data_path_labels)

    if np.max(pre) > 1.5:
        print("transform to softmax vector")
        pre = manual_softmax_row(pre)
        # import scipy
        # pre = scipy.special.softmax(pre,axis=1)

    tripro_cover = TriProCover()

    sp_c_arr, sp_v_arr, cov_num_arr_list, no_cov_num_arr_list = tripro_cover.cal_triangle_cov(pre, labels, nb_classes,
                                                                                              deep_num,
                                                                                              by_deep_num=True)

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 避免负号显示问题
    cov_split = list(zip(*cov_num_arr_list))
    no_cov_split = list(zip(*no_cov_num_arr_list))
    sp_c_str_arr = [num_to_str(x, 5) for x in sp_c_arr]

    for i in range(4):
        cov_data = cov_split[i]
        no_cov_data = no_cov_split[i]

        merged_data = [item for pair in zip(cov_data, no_cov_data) for item in pair]
        # 计算横坐标的累计和，用于折线的延展
        x_data = np.cumsum(merged_data).tolist()
        x_data.insert(0, 0)
        y_data = [item for pair in zip([1] * len(cov_data), [0] * len(cov_data)) for item in pair]
        y_data.append(y_data[len(y_data) - 1])
        # 创建单独的图
        plt.figure(figsize=(8, 6))
        for j in range(len(y_data) - 1):
            # 如果 y = 1，使用红色；如果 y = 0，使用蓝色
            color = 'red' if y_data[j] == 1 else 'blue'

            plt.hlines(y=y_data[j], xmin=x_data[j], xmax=x_data[j + 1], color=color, linewidth=2)

            # 画竖线：在每个变化点处画黑色虚线
            plt.vlines(x=x_data[j + 1], ymin=min(y_data[j], y_data[j + 1]), ymax=max(y_data[j], y_data[j + 1]),
                       color='black', linestyle='--', linewidth=0.5)

        # 添加图例，标题和坐标轴标签
        plt.title(f"Model {model_path}, Coverage in depth {i}, {sp_c_str_arr[i]}", fontsize=14)
        plt.xlabel("Number of regions", fontsize=12)
        plt.ylabel("Coverage state \n1 is coveraged", fontsize=12)
        # 设置y轴的范围和刻度
        plt.ylim(-0.5, 1.5)
        plt.yticks([0, 1])

        os.makedirs(save_path, exist_ok=True)
        img_path = os.path.join(save_path, f'coverage_{i + 1}.png')

        # 保存图像
        plt.savefig(img_path)

    sp_c_str_arr = [num_to_str(x, 5) for x in sp_c_arr]
    tb = prettytable.PrettyTable()
    tb.field_names = ["Deep", "Diversity"]
    print("=" * 100)
    print(f"Data Diverstiy: {sp_c_str_arr[-1]}, Model: {model_path}, Data: {data_path}")
    print("Detail Result")
    for i, ratio in enumerate(sp_c_str_arr):
        tb.add_row([i + 1, ratio])
    #     print("ori data. deep: {} pt coverage: {}".format(i, ratio))
    # color_print("deep {}, pt coverage: {}".format(split_num, sp_c_str_arr[-1]), "blue")
    print(tb)
    return tb


# a demo for pt
if __name__ == '__main__':
    #     try:
    # python tools.py --split_num 4 --class_num 4 --data_path "mnist.zip" --model_path "mnist_LeNet5/model_mnist_LeNet5.hdf5"
    from c2net.context import prepare
    from c2net.context import upload_output
    import os
    import argparse
    import zipfile

    c2net_context = prepare()
    datasetPath = c2net_context.dataset_path
    modelPath = c2net_context.pretrain_model_path
    outputPath = c2net_context.output_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_num")
    parser.add_argument("--class_num")
    parser.add_argument("--data_path")
    parser.add_argument("--model_path")
    args, unknown = parser.parse_known_args()
    # print(args)
    # print("+++++")
    # print(unknown)
    try:
        split_num = int(args.split_num)
        class_num = int(args.class_num)
        _data_path = args.data_path
        _model_path = args.model_path
        params_path = os.path.join(outputPath, "params.txt")
        res = f"split_num {split_num}, class_num {class_num}, data_path {_data_path}, model_path {_model_path}"
        with open(params_path, "w") as f:
            f.writelines(res)
    except:
        try:
            split_num = unknown[1]
            class_num = unknown[3]
            _data_path = unknown[5]
            _model_path = unknown[7]
        except:
            split_num = unknown[1].split("=")[1].replace("\'", "")
            class_num = unknown[2].split("=")[1].replace("\'", "")
            _data_path = unknown[3].split("=")[1].replace("\'", "")
            _model_path = unknown[4].split("=")[1].replace("\'", "")

    data_path = os.path.join(datasetPath, _data_path)
    model_path = os.path.join(modelPath, _model_path)

    f = zipfile.ZipFile(data_path, 'r')
    for file in f.namelist():
        f.extract(file, datasetPath)
    f.close()

    save_path = os.path.join(outputPath, "images")

    res = meature(split_num, class_num, datasetPath, model_path, save_path)

    result_path = os.path.join(outputPath, "result.txt")
    with open(result_path, "w") as f:
        f.writelines(str(res))
    upload_output()
#    except Exception as e:
#         print("cenet error" + e)
#         # python tools.py --split_num 4 --class_num 4 --data_path "data/mnist" --model_path "model/model_mnist_LeNet5.hdf5"
#         Fire(component=meature)