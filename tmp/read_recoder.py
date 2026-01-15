
import pickle
from typing import Dict
# with open('/home/lzq/experiment_datatset/inputs/scene_data.pkl', 'rb') as f:
#     data: Dict[int, Dict] = pickle.load(f)

# print(data)
import numpy as np

# 假设你的文件叫做 data.npy
data = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town1/obstacles_y.npy',allow_pickle=True)
# 输出数据看一眼
print(data)
print(type(data))
print(data.shape)


