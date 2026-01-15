import pickle

# 读取 pickle 文件
pickle_file = '/media/lzq/D/lzq/pylot_test/pylot/error_seeds/crash_seeds/0_90_3.pickle'

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# 打印数据类型和基本信息
print(f"数据类型: {type(data)}")
print(f"\n数据内容:")
print(data)

# 如果是字典，打印键
if isinstance(data, dict):
    print(f"\n字典的键: {list(data.keys())}")
    for key in data.keys():
        print(f"  - {key}: {type(data[key])}")

# 如果是列表，打印长度和前几个元素
if isinstance(data, list):
    print(f"\n列表长度: {len(data)}")
    if len(data) > 0:
        print(f"第一个元素类型: {type(data[0])}")
        print(f"前3个元素: {data[:3]}")