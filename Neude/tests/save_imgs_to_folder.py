import numpy as np
from PIL import Image
import os
output_folder = 'images'
os.makedirs(output_folder, exist_ok=True)

images_array = np.load("/home/lzq/train_x.npy")
output_folder = '/media/lzq/D/lzq/fuzz_tool/demo/input_images_datax'
os.makedirs(output_folder, exist_ok=True) 
for i in range(1000):
    img = Image.fromarray(images_array[i])  # 将 NumPy 数组转换为图像
    name = "{:03}".format(i)
    img.save(os.path.join(output_folder, f'image_{name}.png'))  # 保存为 PNG 格式
