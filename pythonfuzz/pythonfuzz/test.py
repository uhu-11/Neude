import os
import pickle
import sys
sys.path.append('/media/lzq/D/lzq/pylot_test/pythonfuzz')
from pythonfuzz import Seed
if __name__ == '__main__':
        # 文件夹路径
    folder_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp21/local_seeds_pool'

    # 遍历文件夹中的所有 pickle 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl') or filename.endswith('.pickle'):
            file_path = os.path.join(folder_path, filename)

            # 读取 pickle 文件
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            # print(data.params.keys())
            data.params.pop('perfect_depth_estimation')
            data.params.pop('perfect_segmentation')
            data.params.pop('log_detector_output')
            data.params.pop('log_lane_detection_camera')
            data.params.pop('log_traffic_light_detector_output')
            print(data.params.keys())
            # # 在这里对 data 进行修改
            # # 例如，如果 data 是 dict，你可以这样修改：
            # if isinstance(data, dict):
            #     data['new_key'] = 'new_value'

            # # 保存回文件（可以选择覆盖或另存为新文件）
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

            # print(f"{filename} 处理完成")
