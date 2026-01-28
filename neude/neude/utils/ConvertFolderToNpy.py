import os
import numpy as np
from PIL import Image
import logging

# 设置 PIL 的日志级别
logging.getLogger("PIL").setLevel(logging.ERROR)

def load_images_from_folder(folder):
    images = []
    
    # 遍历文件夹中的所有文件
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        
        # 只处理图片文件
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img = Image.open(img_path)  # 打开图片
            img = img.convert('RGB')  # 确保是 RGB 格式
            # img = img.resize(target_size)  # 调整图片大小
            img_array = np.array(img)  # 转换为 NumPy 数组
            images.append(img_array)  # 添加到列表中

    # 将列表转换为 NumPy 数组
    images_array = np.array(images)
    return images_array

if __name__ == '__main__':
    for filename in sorted(os.listdir("/media/lzq/D/lzq/pylot_test/pylot/input_datax")):
        print(filename)