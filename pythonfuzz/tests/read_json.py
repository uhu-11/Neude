# import json
# import os
# from pathlib import Path
# def process_data(data):
#     #这是一个种子的groundtruth

#     result = []
#     for idx, item in enumerate(data):
#         m = {}
#         if 'red' in item[0]:
#             m['label'] = 3
#         elif 'green' in item[0]:
#             m['label'] = 1
#         elif 'yellow' in item[0]:
#             m['label'] = 2
#         else:
#             m['label'] = 4
#         m['box'] = [item[3][0][0], item[3][0][1], item[3][1][0], item[3][1][1]]
#         result.append(m)
        
#     return result

# def read_json_files(directory):
#     """按字典序读取json文件"""
#     # 获取目录下所有json文件
#     json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
#     # 简单的字典序排序
#     json_files.sort()
    
#     data_list = []
#     for filename in json_files:
#         file_path = os.path.join(directory, filename)
#         try:
#             with open(file_path, 'r') as f:
#                 data = json.load(f)
#                 processed_data = process_data(data)
#                 print(filename, processed_data)
#                 data_list.append(processed_data)
#                 print(f"Successfully read: {filename}")
#        except Exception as e:
#             print(f"Error reading {filename}: {str(e)}")
    
#     return data_list
# import numpy as np
# def main():
#     # 指定目录路径
#     directory = "/media/lzq/D/lzq/pylot_test/pylot/yjson"
    
#     # 读取所有json文件
#     json_data = read_json_files(directory)
    
#     # 打印读取的文件信息
#     print(f"\nTotal files read: {len(json_data)}")
    
#     # 打印前几个文件的信息作为示例
#     print("\nFirst few files:")
#     print(json_data)
#     # np.save('light_y.npy', json_data)






# #for 物体检测
# cars = [
#     "dodge",
#     "ford",
#     "lincoln",
#     "mercedes",
#     "mini",
#     "nissan",
#     "audi",
#     "BMW",
#     "chevrolet",
#     "citroen",
#     "jeep",
#     "micro",
#     "seat",
#     "tesla",
#     "toyota"
# ]

# Motorcycle = ['Harley Davidson',
# 'Kawasaki',
# 'Vespa',
# 'Yamaha']

# Bicycle = [
#     "BH",
#     "Diamondback",
#     "Gazelle"
# ]
# def obstacle_read_json_files(directory):
#     """按字典序读取json文件"""
#     # 获取目录下所有json文件
#     json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
#     # 简单的字典序排序
#     json_files.sort()
    
#     data_list = []
#     for filename in json_files:
#         file_path = os.path.join(directory, filename)
#         try:
#             with open(file_path, 'r') as f:
#                 data = json.load(f)
#                 processed_data = obstacle_process_data(data)
#                 print(filename, processed_data)
#                 data_list.append(processed_data)
#                 print(f"Successfully read: {filename}")
#        except Exception as e:
#             print(f"Error reading {filename}: {str(e)}")
    
#     return data_list

# def obstacle_process_data(data):
#     #这是一个种子的groundtruth

#     result = []
#     for idx, item in enumerate(data):
#         print(item)
#         m = {}
#         if 'speed limit 30' in item[0]:
#             m['label'] = 5
#         elif 'speed limit 60' in item[0]:
#             m['label'] = 6
#         elif 'speed limit 90' in item[0]:
#             m['label'] = 7
#         elif "person" in item[0]:
#             m['label'] = 2
#         elif 'vehicle' in item[0]:
#             if any(car.lower() in item[1].lower() for car in cars):
#                 m['label'] = 1
#             elif any(moto.lower() in item[1].lower() for moto in Motorcycle):
#                 m['label'] = 3
#             elif any(bicycle.lower() in item[1].lower() for bicycle in Bicycle):
#                 m['label'] = 4

#         m['box'] = [item[3][0][0], item[3][0][1], item[3][1][0], item[3][1][1]]
#         result.append(m)
        
#     return result

# import numpy as np
# def obs_main():
#     # 指定目录路径
#     directory = "/media/lzq/D/lzq/pylot_test/pylot/obstaclesjson"
    
#     # 读取所有json文件
#     json_data = obstacle_read_json_files(directory)
    
#     # 打印读取的文件信息
#     print(f"\nTotal files read: {len(json_data)}")
    
#     # 打印前几个文件的信息作为示例
#     print("\nFirst few files:")
#     print(json_data)
#     np.save('obs_y.npy', json_data)




import os
import json
import shutil

def process_data(data):
    """处理 JSON 数据"""
    result = []
    for item in data:
        m = {}
        if 'red' in item[0]:
            m['label'] = 3
        elif 'green' in item[0]:
            m['label'] = 1
        elif 'yellow' in item[0]:
            m['label'] = 2
        else:
            m['label'] = 4
        m['box'] = [item[3][0][0], item[3][0][1], item[3][1][0], item[3][1][1]]
        result.append(m)
        
    return result
import re
def process_images_and_json(image_dir, json_dir, copy_path):
    """处理图片和 JSON 文件"""
    if not os.path.exists(copy_path):
        os.makedirs(copy_path)  # 创建目标路径

    # 获取所有 JSON 文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    json_files.sort()  # 按字典序排序
    processed_data_list = []
    i=0

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        match = re.search(r'(\d+)', json_file)
        number=match.group(1)
        image_path = os.path.join(image_dir, f'traffic-light-{number}.png')  

        # try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
            if data:  # 只处理非空 JSON
                # 复制图片到 copy_path
                
                # 处理 JSON 数据
                processed_data = process_data(data)
                if os.path.exists(image_path):
                    print(i, image_path)
                    shutil.copy(image_path, copy_path)
                    print(f"Copied image: {image_path} → {copy_path}")
                else:
                    
                    print(f"Warning: Image not found for {image_path}")
                    continue

                processed_data_list.append(processed_data)
                print(f"Processed JSON: {json_file}")
                
                i+=1
                if i==250:
                        break

        # except Exception as e:
        #     print(f"Error processing {json_file}: {e}")

    return processed_data_list

# import numpy as np
# def main():
#     # 指定目录路径
#     image_dir = '/home/lzq/lights2/traffic-light'
#     json_dir = '/home/lzq/lights2/tl-bboxes'
#     copy_path='/home/lzq/traffic_data_set'
#     # 读取所有json文件
#     json_data = process_images_and_json(image_dir,json_dir,copy_path)

    
#     # 打印读取的文件信息
#     print(f"\nTotal files read: {len(json_data)}")
    
#     # 打印前几个文件的信息作为示例
#     print("\nFirst few files:")
#     # print(json_data)
#     np.save('/home/lzq/lights2/light_y.npy', json_data)
# if __name__ == "__main__":

#     main()




#for 物体检测
cars = [
    "dodge",
    "ford",
    "lincoln",
    "mercedes",
    "mini",
    "nissan",
    "audi",
    "BMW",
    "chevrolet",
    "citroen",
    "jeep",
    "micro",
    "seat",
    "tesla",
    "toyota"
]

Motorcycle = ['Harley Davidson',
'Kawasaki',
'Vespa',
'Yamaha']

Bicycle = [
    "BH",
    "Diamondback",
    "Gazelle"
]
import re
def process_images_and_json(image_dir, json_dir, copy_path):
    """处理图片和 JSON 文件"""
    if not os.path.exists(copy_path):
        os.makedirs(copy_path)  # 创建目标路径

    # 获取所有 JSON 文件并按字典序排序
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    json_files.sort()  # 按字典序排序

    # 找到 'bboxes-00223348.json' 的索引
    start_index = json_files.index('bboxes-00170225.json')
    
    processed_data_list = []
    i = 0

    # 从 start_index 开始读取后续的 100 个非空 JSON 文件
    for json_file in json_files[start_index:start_index + 100]:
        json_path = os.path.join(json_dir, json_file)
        match = re.search(r'(\d+)', json_file)
        
        if match:
            number = match.group(1)
            image_path = os.path.join(image_dir, f'center-{number}.png')  

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                    if data:  # 只处理非空 JSON
                        # 复制图片到 copy_path
                        processed_data = obstacle_process_data(data)
                        if os.path.exists(image_path):
                            shutil.copy(image_path, copy_path)
                            print(f"Copied image: {image_path} → {copy_path}")
                        else:
                            print(f"Warning: Image not found for {image_path}")
                            continue
                        processed_data_list.append(processed_data)
                        print(f"Processed JSON: {json_file}")
                        
                        i += 1
                        if i == 100:  # 只处理 100 个文件
                            break

            except Exception as e:
                print(f"Error processing {json_file}: {e}")

    return processed_data_list

def obstacle_process_data(data):
    #这是一个种子的groundtruth

    result = []
    for idx, item in enumerate(data):
        # print(item)
        m = {}
        if 'speed limit 30' in item[0]:
            m['label'] = 5
        elif 'speed limit 60' in item[0]:
            m['label'] = 6
        elif 'speed limit 90' in item[0]:
            m['label'] = 7
        elif "person" in item[0]:
            m['label'] = 2
        elif 'vehicle' in item[0]:
            if any(car.lower() in item[1].lower() for car in cars):
                m['label'] = 1
            elif any(moto.lower() in item[1].lower() for moto in Motorcycle):
                m['label'] = 3
            elif any(bicycle.lower() in item[1].lower() for bicycle in Bicycle):
                m['label'] = 4
        if 'speed limit' in item[0]:
            m['box'] = [item[1][0][0], item[1][0][1], item[1][1][0], item[1][1][1]]
        else:
            m['box'] = [item[3][0][0], item[3][0][1], item[3][1][0], item[3][1][1]]
        result.append(m)
        
    return result


import numpy as np
def main():
    # 指定目录路径
    image_dir = '/home/lzq/experiment_datatset/all_gathered_dataset/town1/center'
    json_dir = '/home/lzq/experiment_datatset/all_gathered_dataset/town1/bboxes'
    copy_path='/home/lzq/experiment_datatset/fuzz_test_dataset/town1/obstacles_dataset_datax'
    # 读取所有json文件
    json_data = process_images_and_json(image_dir,json_dir,copy_path)

    
    # 打印读取的文件信息
    print(f"\nTotal files read: {len(json_data)}")
    
    # 打印前几个文件的信息作为示例
    print("\nFirst few files:")
    # print(json_data)
    np.save('/home/lzq/experiment_datatset/fuzz_test_dataset/town1/obstacles_y.npy', json_data)
if __name__ == "__main__":

    main()


