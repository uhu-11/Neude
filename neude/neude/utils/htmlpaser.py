from bs4 import BeautifulSoup
import os
from neude.config import EXCLUDE_FILE

global_line_status_vector_map = {}
global_keys=[]
'''
def line_coverage():
    with open("index.html", "r", encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    rows = soup.select('main table tbody tr')
    all_func_lines = 0
    all_missed_lines = 0
    for row in rows:
        # 获取所有<td>元素
        cells = row.find_all('td')
        # 提取第三个和第四个<td>的内容
        all_lines = cells[2].text
        missed_lines = cells[3].text
        all_func_lines = all_func_lines + all_lines
        all_missed_lines = all_missed_lines + missed_lines

    coverage_rate = (all_func_lines - all_missed_lines) / all_func_lines
    return coverage_rate
'''
#测试imageio的函数，几个用例能达到最大覆盖，有没有带模型的接口

def coveraged_line(folder_path):
    cur_line_status_vector_map = {}
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否以 '_py.html' 结尾
            if file.endswith('_py.html'):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding='utf-8') as f:
                    html = f.read()
                soup = BeautifulSoup(html, 'html.parser')
                
                # 1. 从 title 中获取文件名
                title_tag = soup.find('title')
                if title_tag:
                    title_text = title_tag.get_text()
                    if "Coverage for" in title_text:
                        file_name = title_text.split("Coverage for ")[1].split(":")[0].strip()

                # 2. 获取所有函数及其行号
                functions = []
                p_tags = soup.find('main', id='source').find_all('p')

                for idx, p_tag in enumerate(p_tags):
                    if p_tag.find('span', string='def'):
                        start_line = int(p_tag.find('a')['id'][1:]) + 1  # 下一行作为起始行
                        function_name = p_tag.find('span', class_='nam').text.strip()

                        # 默认结束行为起始行
                        end_line = start_line

                        # 找到下一个函数的位置
                        next_func_idx = None
                        for next_idx in range(idx + 1, len(p_tags)):
                            if p_tags[next_idx].find('span', string='def'):
                                next_func_idx = next_idx
                                break

                        # 获取当前函数的所有内容行
                        candidate_lines = p_tags[idx + 1 : next_func_idx] if next_func_idx else p_tags[idx + 1 :]

                        # 筛选出“实际代码”行，取其中最后一行的行号作为 end_line
                        for tag in reversed(candidate_lines):
                            if is_actual_code(tag):
                                end_line = int(tag.find('a')['id'][1:])
                                break
                        else:
                            end_line = start_line  # fallback，如果没找到实际代码

                        functions.append({
                            'function_name': function_name,
                            'start_line': start_line,
                            'end_line': end_line
                        })

                # 3. 获取所有 class 为 'mis show_mis' 或 'mis' 的 p 标签行号
                mis_lines = []
                mis_tags = soup.find_all('p', class_=['mis show_mis', 'mis'])
                for p in mis_tags:
                    if 'coverage_decorator' not in p.get_text():
                        line_number = p.find('a')['id'][1:]
                        mis_lines.append(int(line_number))

                # 4. 获取所有 class 为 'run show_run' 或 'run' 的 p 标签行号
                run_lines = []
                run_tags = soup.find_all('p', class_=['run show_run', 'run'])
                for p in run_tags:
                    if 'coverage_decorator' not in p.get_text():
                        line_number = p.find('a')['id'][1:]
                        run_lines.append(int(line_number))

                # 5. 获取所有 class 为 'pln' 的 p 标签行号
                pln_lines = []
                pln_tags = soup.find_all('p', class_='pln')
                for p in pln_tags:
                    line_number = p.find('a')['id'][1:]
                    pln_lines.append(int(line_number))

                # 遍历 functions 列表，生成覆盖向量
                for func in functions:
                    func_name = func['function_name']
                    start_line = func['start_line']
                    end_line = func['end_line']
                    vector = []

                    for line_num in range(start_line, end_line + 1):
                        if line_num in mis_lines:
                            vector.append(0)  # 未覆盖
                        elif line_num in run_lines:
                            vector.append(1)  # 已覆盖
                        elif line_num in pln_lines:
                            vector.append(-1)  # 空行或注释
                        else:
                            vector.append(-1)  # 默认处理
                    key = f"{file_name}_{func_name}"

                    # if vector[0]==1 and vector.count(1)<2:
                    #     # print(key)
                    #     continue
                    is_exclude_file = False
                    for file in EXCLUDE_FILE:
                        if file in file_name:
                            is_exclude_file=True
                            break
                    if 'pythonfuzz' not in file_name and not is_exclude_file:
                        cur_line_status_vector_map[key] = vector
                    
                        if key in global_line_status_vector_map:
                            global_line_status_vector_map[key] = merge_vectors(global_line_status_vector_map[key], vector)
                        else:
                            global_keys.append(key)
                            global_line_status_vector_map[key] = vector
    #print("global_line_status_vector_map", global_line_status_vector_map)
    # print(global_line_status_vector_map)
    # print(global_line_status_vector_map.keys())
    cur_seed_vector=[]
    for i in range(len(global_keys)-1,-1,-1):
        if global_keys[i] in cur_line_status_vector_map:
            cur_seed_vector = cur_line_status_vector_map[global_keys[i]] + cur_seed_vector
    for k in cur_line_status_vector_map.keys():
        if k not in global_keys:
            cur_seed_vector = cur_line_status_vector_map[k] + cur_seed_vector
    # cur_seed_vector = [item for sublist in cur_line_status_vector_map.values() for item in sublist]
    total_coverage_rate, total_1_count, total_1_and_0_count = calculate_coverage(global_line_status_vector_map)
    return total_coverage_rate, total_1_count, total_1_and_0_count, cur_seed_vector, global_line_status_vector_map, cur_line_status_vector_map, global_keys



def is_comment_or_empty(text):
    # 判断是否是注释行或者空行
    stripped_text = text.strip()
    return stripped_text.startswith('#') or stripped_text == ''


# 获取行内容的状态，1表示高亮，-1表示空行或注释，0表示正常行
def get_line_status(soup, line_num, highlighted_lines):
    line_tag = soup.find('a', id=f't{line_num}')
    if line_tag is None:
        return -1  # 如果找不到该行，默认为空行
    text = line_tag.next_sibling.get_text().strip() if line_tag.next_sibling else ""

    if line_num in highlighted_lines:
        return 0  # 高亮行, 0表示未覆盖
    elif is_comment_or_empty(text):
        return -1  # 注释或空行
    else:
        return 1  # 正常行

def merge_vectors(vector1, vector2):
    merged_vector = []
    for v1, v2 in zip(vector1, vector2):
        if v1 == 1 or v2 == 1:
            merged_vector.append(1)
        elif v1 == 0 and v2 == 0:
            merged_vector.append(0)
        else:
            merged_vector.append(-1)
    return merged_vector

def calculate_coverage(function_vectors):
    total_1_count = 0  # 统计所有向量中1的总数
    total_1_and_0_count = 0  # 统计所有向量中1和0的总数

    # 遍历函数状态向量map
    for func_name, vector in function_vectors.items():
        # if 1 not in vector:
        #     continue
        # # 遍历每个向量中的值
        for i, value in enumerate(vector):
            # if i==0 and value==0:
            #     continue
            if value == 1:
                total_1_count += 1
            if value == 1 or value == 0:
                total_1_and_0_count += 1

    # 计算覆盖率：1的总数 / (1和0的总数)
    if total_1_and_0_count == 0:
        return 0  # 避免除以0的情况
    coverage = total_1_count / total_1_and_0_count
    return coverage, total_1_count, total_1_and_0_count

def is_actual_code(p_tag):
    text = p_tag.get_text(strip=True)
    return text and not text.startswith('@') and not text.startswith('#')
if __name__ == '__main__':
    for root, dirs, files in os.walk('/home/lzq/test_reslut/obstacle/covhtml'):
        for dir_name in dirs:
            print(os.path.join(root, dir_name)) 
            total_coverage_rate, total_1_count, total_1_and_0_count, _=coveraged_line(os.path.join(root, dir_name))
            print(total_coverage_rate, total_1_count, total_1_and_0_count)