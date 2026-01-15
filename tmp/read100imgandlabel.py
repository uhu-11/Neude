'''
不要用这个
'''

# import os
# import json
# import numpy as np
# import shutil

# def process_files(json_folder, image_folder, output_npy_file, output_image_folder):
#     # 获取所有 JSON 文件并按字典序排序
#     json_files = sorted([f for f in os.listdir(json_folder) if f.endswith('.json')])
    
#     # 找到 'bboxes-00223348.json' 的索引
#     start_index = json_files.index('bboxes-00170225.json')
    
#     # 读取后续 100 个 JSON 文件
#     json_data_list = []
#     for i in range(start_index, min(start_index + 100, len(json_files))):
#         json_file_path = os.path.join(json_folder, json_files[i])
#         with open(json_file_path, 'r') as f:
#             json_data = json.load(f)
#             json_data_list.append(json_data)
            
#             # 获取数字部分
#             number_part = json_files[i].split('-')[1].split('.')[0]
#             image_file_name = f'center-{number_part}.png'
#             image_source_path = os.path.join(image_folder, image_file_name)
            
#             # 复制对应的图片
#             if os.path.exists(image_source_path):
#                 shutil.copy(image_source_path, output_image_folder)
#             else:
#                 print(f"Image {image_file_name} does not exist.")

#     # 保存 JSON 数据到 .npy 文件
#     np.save(output_npy_file, json_data_list)
#     print(f"Saved JSON data to {output_npy_file}")
#     print(json_data_list)

# # 使用示例
# json_folder = '/home/lzq/experiment_datatset/all_gathered_dataset/town1/bboxes'
# image_folder = '/home/lzq/experiment_datatset/all_gathered_dataset/town1/center'
# output_npy_file = '/home/lzq/experiment_datatset/fuzz_test_dataset/town1/obstacles_yy.npy'
# output_image_folder = '/home/lzq/experiment_datatset/fuzz_test_dataset/town1/obstacles_dataset_dataxx'  # 目标文件夹

# # 创建目标文件夹（如果不存在）
# os.makedirs(output_image_folder, exist_ok=True)

# process_files(json_folder, image_folder, output_npy_file, output_image_folder)

