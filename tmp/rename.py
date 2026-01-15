import os
import re

def rename_bbox_files(folder_path):
    # 获取文件列表
    files = os.listdir(folder_path)

    for file in files:
        # 使用正则表达式匹配文件名
        match = re.match(r'traffic-light-(\d{1,6})\.png', file)
        if match:
            # 提取数字部分并补零
            number = match.group(1)
            new_number = number.zfill(8)  # 补零到8位
            new_file_name = f'traffic-light-{new_number}.png'
            
            # 构造完整的旧文件路径和新文件路径
            old_file_path = os.path.join(folder_path, file)
            new_file_path = os.path.join(folder_path, new_file_name)
            
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {file} to {new_file_name}')

# 使用示例
folder_path = '/home/lzq/experiment_datatset/all_gathered_dataset/town4/traffic-light'  # 替换为你的文件夹路径
rename_bbox_files(folder_path)


# import json

# # 读取 JSON 文件
# file_path = '/home/lzq/result/datas/5.json'  # 替换为你的文件路径

# with open(file_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # 检查数据类型
# if isinstance(data, dict):
#     print("Loaded data is a dictionary.")
#     print(data)  # 打印字典内容
# else:
#     print("Loaded data is not a dictionary.")

    


