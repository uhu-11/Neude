import os
import numpy as np
from bs4 import BeautifulSoup

def parse_html_file(file_path):
    """
    解析HTML文件，返回文件名和覆盖向量
    返回: (file_name, vector_dict) 其中vector_dict的key是行号，value是覆盖状态
    """
    with open(file_path, "r", encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    
    # 1. 从 title 中获取文件名file_name
    title_tag = soup.find('title')
    file_name = None
    if title_tag:
        title_text = title_tag.get_text()
        if "Coverage for" in title_text:
            file_name = title_text.split("Coverage for ")[1].split(":")[0].strip()
    
    if not file_name:
        return None, None
    
    # 2. 获取main标签下的所有p标签（代码行）
    main_tag = soup.find('main', id='source')
    if not main_tag:
        return None, None
    
    p_tags = main_tag.find_all('p')
    
    # 3. 创建向量字典，key为行号，value为覆盖状态
    vector_dict = {}
    
    for p_tag in p_tags:
        # 获取行号
        a_tag = p_tag.find('a', id=True)
        if not a_tag:
            continue
        
        line_id = a_tag.get('id', '')
        if not line_id.startswith('t'):
            continue
        
        try:
            line_number = int(line_id[1:])  # 去掉't'前缀
        except ValueError:
            continue
        
        # 根据class判断覆盖状态
        classes = p_tag.get('class', [])
        if isinstance(classes, str):
            classes = [classes]
        
        # 判断覆盖状态
        # 优先级：run > mis > pln
        if 'run' in classes:
            # 已覆盖（包括show_run和par run等情况）
            vector_dict[line_number] = 1
        elif 'mis' in classes:
            # 未覆盖（包括show_mis）
            vector_dict[line_number] = 0
        elif 'pln' in classes:
            # 空行或注释
            vector_dict[line_number] = -1
        else:
            # 默认处理为空行或注释
            vector_dict[line_number] = -1
    
    return file_name, vector_dict


def process_iteration(iteration):
    """
    处理单个iteration目录
    """
    iteration_dir = f"pylot/covhtml/{iteration}"
    
    if not os.path.exists(iteration_dir):
        print(f"目录不存在: {iteration_dir}")
        return None
    
    # 存储所有文件的向量
    all_vectors = {}
    
    # 遍历目录下的所有文件
    for file in os.listdir(iteration_dir):
        # 检查文件是否以 '_py.html' 结尾
        if not file.endswith('_py.html'):
            continue
        
        # 排除文件名包含pythonfuzz的文件
        if 'pythonfuzz' in file:
            continue
        
        file_path = os.path.join(iteration_dir, file)
        
        # 解析HTML文件
        file_name, vector_dict = parse_html_file(file_path)
        
        if file_name and vector_dict:
            # 将vector_dict转换为数组，行号作为索引
            # 找到最大行号
            max_line = max(vector_dict.keys()) if vector_dict else 0
            
            # 创建向量，索引从1开始（0位置不使用）
            vector = np.zeros(max_line + 1, dtype=np.int32)
            for line_num, status in vector_dict.items():
                vector[line_num] = status
            
            # 数组命名为{file_name}_vector
            vector_name = f"{file_name}_vector"
            all_vectors[vector_name] = vector
    
    return all_vectors


def main():
    # 创建输出目录
    output_dir = "/media/lzq/D/lzq/pylot_test/pylot/cov_vector"
    # ori_seeds目录：
    # output_dir = "/media/lzq/D/lzq/pylot_test/pylot/ori_cov_vector"

    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历0-99的iteration
    for iteration in range(100):
        print(f"\n处理 iteration {iteration}...")
        
        # 处理当前iteration
        all_vectors = process_iteration(iteration)
        
        if all_vectors:
            # 保存为npy文件，文件名为{iteration+1}_vector.npy
            output_file = os.path.join(output_dir, f"{iteration+1}_vector.npy")
            
            # 保存字典
            np.save(output_file, all_vectors)
            
            print(f"已保存到: {output_file}")
            print(f"包含 {len(all_vectors)} 个文件的向量")
            
            # 打印文件内容
            print("\n文件内容:")
            for vector_name, vector in all_vectors.items():
                # 统计覆盖情况（排除索引0，因为行号从1开始）
                if len(vector) > 1:
                    # 统计1（覆盖）、0（未覆盖）、-1（空行/注释）的数量
                    covered_lines = np.sum(vector[1:] == 1)  # 覆盖的行数
                    missed_lines = np.sum(vector[1:] == 0)  # 未覆盖的行数
                    empty_lines = np.sum(vector[1:] == -1)  # 空行/注释的行数
                    total_lines = len(vector) - 1  # 总行数（排除索引0）
                    
                    print(f"  {vector_name}: 总行数={total_lines}, 覆盖={covered_lines}, 未覆盖={missed_lines}, 空行/注释={empty_lines}")
                    
                    # 打印前20个有意义的行作为示例（排除-1）
                    meaningful_indices = np.where(vector[1:] != -1)[0] + 1  # +1因为从索引1开始
                    if len(meaningful_indices) > 0:
                        sample_indices = meaningful_indices[:20]
                        sample_values = vector[sample_indices]
                        print(f"    示例行号={sample_indices.tolist()}, 值={sample_values.tolist()}")
                else:
                    print(f"  {vector_name}: 长度为{len(vector)}, 无有效行")
        else:
            print(f"iteration {iteration} 没有找到有效的HTML文件")


if __name__ == "__main__":
    main()
