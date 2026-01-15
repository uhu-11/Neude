import os
import numpy as np

def load_npy_files_from_directory(directory):
    """读取指定目录中的所有 .npy 文件并返回它们的内容"""
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]  # 获取所有 .npy 文件
    npy_files.sort()  # 按字典序排序
    data_list = []  # 用于存储所有文件的内容

    for npy_file in npy_files:
        file_path = os.path.join(directory, npy_file)  # 构造完整的文件路径
        try:
            data = np.load(file_path, allow_pickle=True)  # 读取 .npy 文件
            # dict_list = dict(data.item())
            data_list.append(data.tolist())  # 将数据添加到列表中
            print(f"Loaded {npy_file} with shape {data.shape}")  # 打印文件名和数据形状
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")  # 处理读取错误

    return data_list

# 使用示例
def make_control():

    directory = '/media/lzq/D/lzq/pylot_test/pylot/predictions/control'
    npy_data_list = load_npy_files_from_directory(directory)
    np.save(f'/home/lzq/experiment_datatset/fuzz_test_dataset/town4/controls/town4_traffic_light/control_replay_rs_label_y.npy', npy_data_list)
    print(npy_data_list)
    print(len(npy_data_list))

def make_plan_label_y_npy():

    directory = '/media/lzq/D/lzq/pylot_test/pylot/predictions/planning'
    npy_data_list = load_npy_files_from_directory(directory)
    np.save(f'/home/lzq/experiment_datatset/fuzz_test_dataset/town4/plannings/town4_traffic_light/planning_replay_rs_label_y.npy', npy_data_list)
    print(npy_data_list)
    print(len(npy_data_list))
make_control()
make_plan_label_y_npy()
# npy_data_list 现在包含所有 .npy 文件的内容