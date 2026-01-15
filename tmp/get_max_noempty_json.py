import os
import json

def is_json_file_non_empty(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not bool(data):
                print(filepath)
            return bool(data)  # True if data is not empty
    except Exception:
        return False  # Treat as empty if JSON is invalid or unreadable

def find_longest_non_empty_sequence(json_dir):
    files = sorted(f for f in os.listdir(json_dir) if f.endswith('.json'))
    
    max_len = 0
    max_start = None
    max_end = None

    current_len = 0
    current_start = None

    for file in files:
        full_path = os.path.join(json_dir, file)
        if is_json_file_non_empty(full_path):
            if current_len == 0:
                current_start = file
            current_len += 1
            current_end = file
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
                max_end = current_end
        else:
            current_len = 0
            current_start = None

    return max_len, max_start, max_end

# 示例用法
# 把 'your/json/folder' 改成你的 JSON 文件所在的路径
directory_path = '/home/lzq/experiment_datatset/all_gathered_dataset/town5/bboxes'
length, start, end = find_longest_non_empty_sequence(directory_path)
print(f"最长非空序列长度: {length}")
print(f"起始文件: {start}")
print(f"结束文件: {end}")
