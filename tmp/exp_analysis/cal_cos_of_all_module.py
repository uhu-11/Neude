
import os
import numpy as np
import math
from collections import defaultdict
from itertools import combinations


def extract_error_signature(error_content):
    """从错误信息中提取错误签名（用于识别相同错误）"""
    lines = error_content.strip().split('\n')
    # 提取最后一行作为主要错误类型
    if lines:
        last_line = lines[-1].strip()
        # 提取错误类型和关键信息
        error_type = last_line.split(':')[0] if ':' in last_line else last_line
        return error_type, last_line
    return None, None


def read_error_info(file_path):
    """读取错误信息文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"读取错误信息文件失败 {file_path}: {e}")
        return None


def read_npy_file(file_path):
    """读取npy文件并返回字典"""
    try:
        data = np.load(file_path, allow_pickle=True)
        # 如果保存的是字典（通过np.save保存的字典），需要调用.item()
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        return data
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None


def merge_vectors(dict1, dict2, exclude_keyword='pythonfuzz'):
    """
    整合两个npy文件中的数组，排除包含指定关键字的数组
    
    参数:
        dict1: 第一个npy文件的字典
        dict2: 第二个npy文件的字典
        exclude_keyword: 要排除的关键字（默认'pythonfuzz'）
    
    返回:
        merged_vec1: 整合后的第一个向量
        merged_vec2: 整合后的第二个向量
        array_names: 数组名称列表
        missing_in_dict1: 在dict1中缺失的数组名称列表
        missing_in_dict2: 在dict2中缺失的数组名称列表
    """
    # 过滤掉包含exclude_keyword的数组
    filtered_dict1 = {k: v for k, v in dict1.items() if exclude_keyword not in k}
    filtered_dict2 = {k: v for k, v in dict2.items() if exclude_keyword not in k}
    
    # 获取所有唯一的数组名称，并排序以确保顺序一致
    all_keys = sorted(set(list(filtered_dict1.keys()) + list(filtered_dict2.keys())))
    
    missing_in_dict1 = []
    missing_in_dict2 = []
    merged_vec1 = []
    merged_vec2 = []
    
    # 找到每个数组的最大长度
    max_lengths = {}
    for key in all_keys:
        max_len = 0
        if key in filtered_dict1:
            max_len = max(max_len, len(filtered_dict1[key]))
        if key in filtered_dict2:
            max_len = max(max_len, len(filtered_dict2[key]))
        max_lengths[key] = max_len
    
    # 整合数组
    for key in all_keys:
        # 处理dict1中的数组
        if key in filtered_dict1:
            vec1 = filtered_dict1[key]
            # 如果长度不足，用0填充
            if len(vec1) < max_lengths[key]:
                padding = np.zeros(max_lengths[key] - len(vec1), dtype=vec1.dtype)
                vec1 = np.concatenate([vec1, padding])
            merged_vec1.extend(vec1)
        else:
            # 如果缺失，用0填充
            missing_in_dict1.append(key)
            merged_vec1.extend(np.zeros(max_lengths[key], dtype=np.int32))
        
        # 处理dict2中的数组
        if key in filtered_dict2:
            vec2 = filtered_dict2[key]
            # 如果长度不足，用0填充
            if len(vec2) < max_lengths[key]:
                padding = np.zeros(max_lengths[key] - len(vec2), dtype=vec2.dtype)
                vec2 = np.concatenate([vec2, padding])
            merged_vec2.extend(vec2)
        else:
            # 如果缺失，用0填充
            missing_in_dict2.append(key)
            merged_vec2.extend(np.zeros(max_lengths[key], dtype=np.int32))
    
    return np.array(merged_vec1), np.array(merged_vec2), all_keys, missing_in_dict1, missing_in_dict2


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    过滤掉值为-1的位置（-1代表空行和注释，对覆盖向量没有意义）
    """
    # 检查向量长度是否一致
    if len(vec1) != len(vec2):
        return None, f"向量长度不一致: {len(vec1)} vs {len(vec2)}"
    
    # 过滤掉值为-1的位置（如果某个位置在任一向量中为-1，就过滤掉）
    valid_mask = (vec1 != -1) & (vec2 != -1)
    
    # 如果没有有效位置，返回0
    if not np.any(valid_mask):
        return 0.0, "所有位置都为-1，无法计算相似度"
    
    # 提取有效位置的值
    valid_vec1 = vec1[valid_mask]
    valid_vec2 = vec2[valid_mask]
    
    # 计算点积
    dot_product = np.dot(valid_vec1, valid_vec2)
    
    # 计算向量的模
    norm1 = np.linalg.norm(valid_vec1)
    norm2 = np.linalg.norm(valid_vec2)
    
    # 避免除零
    if norm1 == 0 or norm2 == 0:
        return 0.0, "过滤后其中一个向量为零向量"
    
    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
    return similarity, None


def main():
    error_infos_dir = '/media/lzq/D/lzq/pylot_test/pylot/error_infos'
    vectors_dir = '/media/lzq/D/lzq/pylot_test/pylot/cov_vector'
    
    # 按错误签名分组
    error_groups = defaultdict(list)
    
    # 直接从error_infos目录读取所有错误文件，只处理有报错的迭代
    error_iterations = set()
    if os.path.exists(error_infos_dir):
        for filename in os.listdir(error_infos_dir):
            if filename.endswith('.txt'):
                # 从文件名中提取迭代序号，例如从 "2.txt" 提取 "2"
                iteration_num = filename.replace('.txt', '')
                error_iterations.add(iteration_num)
    
    print(f"从错误信息目录中找到 {len(error_iterations)} 个有报错的用例\n")
    
    # 读取错误信息文件并分组
    if os.path.exists(error_infos_dir):
        for iteration_num in error_iterations:
            error_file = os.path.join(error_infos_dir, f"{iteration_num}.txt")
            if os.path.exists(error_file):
                error_content = read_error_info(error_file)
                if error_content:
                    error_type, error_signature = extract_error_signature(error_content)
                    if error_signature:
                        error_groups[error_signature].append(iteration_num)
    
    print(f"找到 {len(error_groups)} 种不同的错误类型\n")
    
    # 收集所有错误组的向量数据
    all_error_vectors = {}  # {error_signature: {iter_num: dict_data}}
    
    # 先读取所有向量数据
    for error_signature, iteration_nums in error_groups.items():
        vectors_data = {}
        for iter_num in iteration_nums:
            vector_file = os.path.join(vectors_dir, f"{iter_num}_array_vector.npy")
            if os.path.exists(vector_file):
                data = read_npy_file(vector_file)
                if data and isinstance(data, dict):
                    vectors_data[iter_num] = data
        if vectors_data:
            all_error_vectors[error_signature] = vectors_data
    
    # 计算相同错误内部的相似度
    print("=" * 80)
    print("计算触发相同错误的用例之间的覆盖向量余弦相似度")
    print("=" * 80)
    
    same_error_similarities = []
    
    for error_signature, iteration_nums in error_groups.items():
        if len(iteration_nums) < 2:
            continue  # 只有一个用例，无法计算相似度
        
        vectors_data = all_error_vectors.get(error_signature, {})
        if len(vectors_data) < 2:
            continue
        
        print(f"\n错误类型: {error_signature}")
        print(f"触发该错误的迭代序号: {sorted([int(x) for x in iteration_nums])}")
        print(f"用例数量: {len(iteration_nums)}\n")
        
        # 计算所有用例对之间的相似度
        for (iter1, iter2) in combinations(sorted(vectors_data.keys(), key=int), 2):
            dict1 = vectors_data[iter1]
            dict2 = vectors_data[iter2]
            
            # 整合向量
            merged_vec1, merged_vec2, array_names, missing1, missing2 = merge_vectors(dict1, dict2)
            
            # 打印整合信息
            print(f"  迭代 {iter1} vs {iter2}:")
            print(f"    整合后的数组数量: {len(array_names)}")
            print(f"    整合后的向量长度: {len(merged_vec1)}")
            if missing1:
                print(f"    在迭代 {iter1} 中缺失的数组: {missing1}")
            if missing2:
                print(f"    在迭代 {iter2} 中缺失的数组: {missing2}")
            
            # 计算余弦相似度（会自动过滤-1值）
            similarity, error_msg = cosine_similarity(merged_vec1, merged_vec2)
            
            if error_msg:
                print(f"    余弦相似度: {error_msg}")
            else:
                # 统计过滤后的有效位置数量
                valid_mask = (merged_vec1 != -1) & (merged_vec2 != -1)
                valid_count = np.sum(valid_mask)
                print(f"    过滤-1后的有效位置数: {valid_count} / {len(merged_vec1)}")
                print(f"    余弦相似度: {similarity:.6f}")
                same_error_similarities.append(similarity)
            print()
    
    # 计算不同错误之间的相似度
    print("\n" + "=" * 80)
    print("计算触发不同错误的用例之间的覆盖向量余弦相似度")
    print("=" * 80)
    
    different_error_similarities = []
    
    error_signatures = list(all_error_vectors.keys())
    for i, error1 in enumerate(error_signatures):
        for error2 in error_signatures[i+1:]:
            vectors1 = all_error_vectors[error1]
            vectors2 = all_error_vectors[error2]
            
            print(f"\n错误类型1: {error1}")
            print(f"  迭代序号: {sorted([int(x) for x in vectors1.keys()], key=int)}")
            print(f"错误类型2: {error2}")
            print(f"  迭代序号: {sorted([int(x) for x in vectors2.keys()], key=int)}")
            
            # 计算所有用例对之间的相似度
            for iter1 in vectors1.keys():
                for iter2 in vectors2.keys():
                    dict1 = vectors1[iter1]
                    dict2 = vectors2[iter2]
                    
                    # 整合向量
                    merged_vec1, merged_vec2, array_names, missing1, missing2 = merge_vectors(dict1, dict2)
                    
                    # 打印整合信息
                    print(f"  迭代 {iter1} vs {iter2}:")
                    print(f"    整合后的数组数量: {len(array_names)}")
                    print(f"    整合后的向量长度: {len(merged_vec1)}")
                    if missing1:
                        print(f"    在迭代 {iter1} 中缺失的数组: {missing1}")
                    if missing2:
                        print(f"    在迭代 {iter2} 中缺失的数组: {missing2}")
                    
                    # 计算余弦相似度（会自动过滤-1值）
                    similarity, error_msg = cosine_similarity(merged_vec1, merged_vec2)
                    
                    if error_msg:
                        print(f"    余弦相似度: {error_msg}")
                    else:
                        # 统计过滤后的有效位置数量
                        valid_mask = (merged_vec1 != -1) & (merged_vec2 != -1)
                        valid_count = np.sum(valid_mask)
                        print(f"    过滤-1后的有效位置数: {valid_count} / {len(merged_vec1)}")
                        print(f"    余弦相似度: {similarity:.6f}")
                        different_error_similarities.append(similarity)
                    print()
    
    # 对比统计
    print("\n" + "=" * 80)
    print("相似度对比统计")
    print("=" * 80)
    
    if same_error_similarities:
        print(f"\n相同错误内部:")
        print(f"  平均相似度: {sum(same_error_similarities) / len(same_error_similarities):.6f}")
        print(f"  最小相似度: {min(same_error_similarities):.6f}")
        print(f"  最大相似度: {max(same_error_similarities):.6f}")
        print(f"  样本数量: {len(same_error_similarities)}")
    
    if different_error_similarities:
        print(f"\n不同错误之间:")
        print(f"  平均相似度: {sum(different_error_similarities) / len(different_error_similarities):.6f}")
        print(f"  最小相似度: {min(different_error_similarities):.6f}")
        print(f"  最大相似度: {max(different_error_similarities):.6f}")
        print(f"  样本数量: {len(different_error_similarities)}")


if __name__ == '__main__':
    main()
