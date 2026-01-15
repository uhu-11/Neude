import json
import csv

# 读取 JSON 文件
# json_file_path = '/home/lzq/test_reslut/会议论文实验结果/exp10/100.json'  # 替换为你的 JSON 文件路径
# csv_file_path = '/home/lzq/test_reslut/会议论文实验结果/exp10/100.csv'  # 替换为你想要保存的 CSV 文件路径

json_file_path = '/home/lzq/result/datas/100.json'  # 替换为你的 JSON 文件路径
csv_file_path = '/home/lzq/result/datas/100.csv'  # 替换为你想要保存的 CSV 文件路径

# 读取 JSON 数据
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 准备 CSV 数据
# 创建 CSV 文件并写入数据
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)

    # 写入列标题
    headers = list(data.keys())  # 添加 'ind' 列标题
    headers.insert(0,'iteration')
    writer.writerow(headers)

    # 写入每一行数据
    for i in range(len(list(data.values())[0])):
        row = [i]  # 每行的第一列是 ind
        for key in data.keys():
            row.append(data[key][i])  # 添加字典中的值
        writer.writerow(row)

print(f"Data has been written to {csv_file_path}.")


# new_ious=[]
# new_dist=[]
# new_sdif=[]
# print(len(data['ious']))
# print(len(data['tdist']))
# print(len(data['steer_diff']))
# l = len(data['ious'])
# i=0
# while i < l:
#     if len(data['ious'][i])!=0:
#         new_ious.append(data['ious'][i])
#         new_dist.append(data['tdist'][i])
#         new_sdif.append(data['steer_diff'][i])
#         i+=1
#         print(data['ious'][i])
#         print(data['tdist'][i])
#         print(data['steer_diff'][i])
#     else:
#         new_ious.append(data['ious'][i])
#         new_dist.append(data['tdist'][i])
#         new_sdif.append(data['steer_diff'][i])
#     i+=1
# print(len(new_ious))
# print(len(new_dist))
# print(len(new_sdif))
# data['ious']=new_ious
# data['tdist']=new_dist
# data['steer_diff']=new_sdif
# for v in data.values():
#     print(len(v))

# with open(f'/home/lzq/test_reslut/会议论文实验结果/exp7/100_true.json', "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)