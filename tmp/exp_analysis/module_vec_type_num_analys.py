'''
遍历每个迭代的覆盖向量，统计perception_vector，planning_vector，control_vector，nac_vector的种类数量，并计算每组神经元覆盖向量的相似性，和组间的相似性。
'''

import os
import pickle
import json
import numpy as np
from collections import defaultdict
from itertools import combinations


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


def read_pickle_file(file_path):
    """读取pickle文件并返回字典"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"读取pickle文件失败 {file_path}: {e}")
        return None


def read_error_data(file_path):
    """读取错误数据JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"读取错误数据文件失败 {file_path}: {e}")
        return None


def extract_error_type(error_file_path):
    """从错误信息文件中提取错误类型"""
    try:
        if not os.path.exists(error_file_path):
            return None
        
        with open(error_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试提取常见的Python错误类型
        error_types = [
            'IndexError', 'ValueError', 'KeyError', 'TypeError', 
            'AttributeError', 'NameError', 'ZeroDivisionError',
            'FileNotFoundError', 'PermissionError', 'OSError',
            'RuntimeError', 'AssertionError', 'ImportError',
            'StopIteration', 'GeneratorExit', 'SystemExit',
            'KeyboardInterrupt', 'Exception'
        ]
        
        for error_type in error_types:
            if error_type in content:
                return error_type
        
        # 如果没有找到标准错误类型，尝试从Traceback中提取
        if 'Traceback' in content:
            lines = content.split('\n')
            for line in lines:
                if ':' in line and any(err in line for err in ['Error', 'Exception']):
                    # 尝试提取错误类型
                    parts = line.split(':')
                    if len(parts) > 0:
                        error_part = parts[0].strip()
                        if any(err in error_part for err in error_types):
                            for err in error_types:
                                if err in error_part:
                                    return err
        
        return 'Unknown'
    except Exception as e:
        return None


def load_code_error_types(error_infos_dir, max_iter=100):
    """加载所有迭代的代码错误类型"""
    error_types_dict = {}  # {iter_num: error_type}
    
    if not os.path.exists(error_infos_dir):
        return error_types_dict
    
    for iter_num in range(1, max_iter + 1):
        error_file = os.path.join(error_infos_dir, f"{iter_num}.txt")
        error_type = extract_error_type(error_file)
        if error_type:
            error_types_dict[str(iter_num)] = error_type
    
    return error_types_dict


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    # 检查向量长度是否一致
    if len(vec1) != len(vec2):
        return None, f"向量长度不一致: {len(vec1)} vs {len(vec2)}"
    
    # 过滤掉值为-1的位置（如果某个位置在任一向量中为-1，就过滤掉）
    # valid_mask = ~((vec1 == -1) & (vec2 == -1))
    
    # 如果没有有效位置，返回0
    # if not np.any(valid_mask):
    #     return 0.0, "所有位置都为-1，无法计算相似度"
    
    # 提取有效位置的值
    # valid_vec1 = vec1[valid_mask]
    # valid_vec2 = vec2[valid_mask]

    valid_vec1 = vec1
    valid_vec2 = vec2
    
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


def arrays_equal(arr1, arr2):
    """判断两个数组是否完全相等"""
    if arr1.shape != arr2.shape:
        return False
    return np.array_equal(arr1, arr2)



def group_module_vectors(module_vectors_dict, similarity_threshold=0.999):
    """
    对模块向量进行分组
    使用精确匹配或高相似度阈值来判断是否属于同一组
    
    返回:
        groups: {group_id: [iteration_numbers]} 字典
        group_representatives: {group_id: representative_vector} 字典
    """
    groups = {}
    group_representatives = {}
    group_id = 0
    
    # 按迭代号排序
    sorted_iterations = sorted(module_vectors_dict.keys(), key=int)
    
    for iter_num in sorted_iterations:
        current_vec = module_vectors_dict[iter_num]
        
        # 查找是否属于已有组
        found_group = False
        for gid, rep_vec in group_representatives.items():
            # 先尝试精确匹配
            if arrays_equal(current_vec, rep_vec):
                groups[gid].append(iter_num)
                found_group = True
                break
            # 如果精确匹配失败，尝试余弦相似度
            else:
                sim, _ = cosine_similarity(current_vec, rep_vec)
                if sim is not None and sim >= similarity_threshold:
                    groups[gid].append(iter_num)
                    found_group = True
                    break
        
        # 如果没有找到匹配的组，创建新组
        if not found_group:
            groups[group_id] = [iter_num]
            group_representatives[group_id] = current_vec
            group_id += 1
    
    return groups, group_representatives


def count_unique_vectors(vectors_list):
    """统计向量列表中有多少种不同的向量"""
    unique_vectors = []
    unique_count = 0
    
    for vec in vectors_list:
        is_unique = True
        for unique_vec in unique_vectors:
            if arrays_equal(vec, unique_vec):
                is_unique = False
                break
        if is_unique:
            unique_vectors.append(vec)
            unique_count += 1
    
    return unique_count


def calculate_group_similarity(vectors_list):
    """
    计算一组向量之间的平均相似度
    如果只有一个向量，返回相似度为1
    """
    if len(vectors_list) == 1:
        return 1.0, 1.0, 1.0, 1.0
    
    if len(vectors_list) < 2:
        return None, None, None, None
    
    similarities = []
    for vec1, vec2 in combinations(vectors_list, 2):
        sim, error_msg = cosine_similarity(vec1, vec2)
        if sim is not None:
            similarities.append(sim)
    
    if not similarities:
        return None, None, None, None
    
    avg_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    median_sim = np.median(similarities)
    
    return avg_sim, min_sim, max_sim, median_sim


def calculate_between_groups_similarity(group1_vectors, group2_vectors):
    """计算两组向量之间的平均相似度"""
    if not group1_vectors or not group2_vectors:
        return None
    
    similarities = []
    for vec1 in group1_vectors:
        for vec2 in group2_vectors:
            sim, error_msg = cosine_similarity(vec1, vec2)
            if sim is not None:
                similarities.append(sim)
    
    if not similarities:
        return None
    
    return np.mean(similarities), np.min(similarities), np.max(similarities), np.median(similarities)


def analyze_module(module_name, module_vectors_dict, nac_vectors_dict, output_file, error_data=None):
    """
    分析一个模块的向量类型和神经元覆盖向量的相似性
    
    参数:
        module_name: 模块名称（如 'perception_vector'）
        module_vectors_dict: {iter_num: module_vector} 字典
        nac_vectors_dict: {iter_num: nac_vector} 字典
        output_file: 输出文件对象
        error_data: 错误数据字典，包含cur_code_errors, cur_model_errors, cur_planning_errors, cur_control_errors
    """
    print(f"\n{'='*80}")
    print(f"分析模块: {module_name}")
    print(f"{'='*80}")
    output_file.write(f"\n{'='*80}\n")
    output_file.write(f"分析模块: {module_name}\n")
    output_file.write(f"{'='*80}\n")
    
    # 1. 对模块向量进行分组
    groups, group_representatives = group_module_vectors(module_vectors_dict)
    
    print(f"\n{module_name} 共有 {len(groups)} 种不同的类型")
    output_file.write(f"\n{module_name} 共有 {len(groups)} 种不同的类型\n")
    
    # 打印每个组的信息
    for group_id, iter_nums in sorted(groups.items()):
        print(f"\n  组 {group_id}: {len(iter_nums)} 个迭代 - {sorted([int(x) for x in iter_nums])}")
        output_file.write(f"\n  组 {group_id}: {len(iter_nums)} 个迭代 - {sorted([int(x) for x in iter_nums])}\n")
    
    # 2. 对每个组，统计神经元覆盖向量的种类数量
    print(f"\n各组神经元覆盖向量统计:")
    output_file.write(f"\n各组神经元覆盖向量统计:\n")
    
    group_nac_vectors = {}  # {group_id: [nac_vectors]}
    
    for group_id, iter_nums in sorted(groups.items()):
        nac_vectors_in_group = []
        for iter_num in iter_nums:
            if iter_num in nac_vectors_dict and nac_vectors_dict[iter_num] is not None:
                nac_vectors_in_group.append(nac_vectors_dict[iter_num])
        
        group_nac_vectors[group_id] = nac_vectors_in_group
        
        if nac_vectors_in_group:
            unique_count = count_unique_vectors(nac_vectors_in_group)
            print(f"  组 {group_id}: {len(nac_vectors_in_group)} 个神经元覆盖向量，其中 {unique_count} 种不同的类型")
            output_file.write(f"  组 {group_id}: {len(nac_vectors_in_group)} 个神经元覆盖向量，其中 {unique_count} 种不同的类型\n")
        else:
            print(f"  组 {group_id}: 没有神经元覆盖向量")
            output_file.write(f"  组 {group_id}: 没有神经元覆盖向量\n")
    
    # 3. 计算每组内神经元覆盖向量的相似性（不输出详细结果）
    all_within_group_similarities = []  # 收集所有组内相似度
    
    for group_id in sorted(groups.keys()):
        nac_vectors = group_nac_vectors.get(group_id, [])
        if len(nac_vectors) >= 1:
            # 收集组内所有两两相似度
            if len(nac_vectors) >= 2:
                for vec1, vec2 in combinations(nac_vectors, 2):
                    sim, _ = cosine_similarity(vec1, vec2)
                    if sim is not None:
                        all_within_group_similarities.append(sim)
            elif len(nac_vectors) == 1:
                # 单个向量，相似度记为1
                all_within_group_similarities.append(1.0)
    
    # 4. 计算组间神经元覆盖向量的相似性（不输出结果）
    all_between_group_similarities = []  # 收集所有组间相似度
    
    group_ids = sorted(groups.keys())
    for i, group1_id in enumerate(group_ids):
        for group2_id in group_ids[i+1:]:
            nac_vectors1 = group_nac_vectors.get(group1_id, [])
            nac_vectors2 = group_nac_vectors.get(group2_id, [])
            
            if nac_vectors1 and nac_vectors2:
                # 收集组间所有两两相似度
                for vec1 in nac_vectors1:
                    for vec2 in nac_vectors2:
                        sim, _ = cosine_similarity(vec1, vec2)
                        if sim is not None:
                            all_between_group_similarities.append(sim)
    
    # 5. 统计每组向量与发生错误之间的关联
    if error_data is not None:
        print(f"\n各组向量与发生错误之间的关联:")
        output_file.write(f"\n各组向量与发生错误之间的关联:\n")
        
        # 获取错误数据
        cur_code_errors = error_data.get('cur_code_errors', [])
        cur_model_errors = error_data.get('cur_model_errors', [])
        cur_planning_errors = error_data.get('cur_planning_errors', [])
        cur_control_errors = error_data.get('cur_control_errors', [])
        
        # 根据模块名称确定使用哪个错误数据
        if module_name == 'total_vector':
            # total_vector需要对所有错误类型进行分类
            error_keys_to_analyze = ['cur_code_errors', 'cur_model_errors', 'cur_planning_errors', 'cur_control_errors']
        else:
            error_key_map = {
                'perception_vector': 'cur_model_errors',
                'planning_vector': 'cur_planning_errors',
                'control_vector': 'cur_control_errors',
            }
            error_keys_to_analyze = [error_key_map.get(module_name)]
        
        for error_key in error_keys_to_analyze:
            if not error_key:
                continue
            
            error_list = error_data.get(error_key, [])
            
            # 对于cur_code_errors，需要细分错误类型
            if error_key == 'cur_code_errors':
                # 加载代码错误类型信息
                error_infos_dir = pylot/error_infos'
                code_error_types = load_code_error_types(error_infos_dir)
                
                # 按错误类型分组统计
                error_type_groups = defaultdict(list)  # {error_type: [iter_nums]}
                
                for group_id, iter_nums in sorted(groups.items()):
                    for iter_num in iter_nums:
                        iter_idx = int(iter_num) - 1
                        if 0 <= iter_idx < len(error_list) and error_list[iter_idx] > 0:
                            error_type = code_error_types.get(iter_num, 'Unknown')
                            error_type_groups[error_type].append((group_id, int(iter_num)))
                
                # 对每种错误类型进行统计
                print(f"\n  错误类型: {error_key} (按错误类型细分)")
                output_file.write(f"\n  错误类型: {error_key} (按错误类型细分)\n")
                
                for error_type in sorted(error_type_groups.keys()):
                    type_items = error_type_groups[error_type]
                    print(f"\n    错误子类型: {error_type} (共 {len(type_items)} 个错误)")
                    output_file.write(f"\n    错误子类型: {error_type} (共 {len(type_items)} 个错误)\n")
                    
                    # 按组统计
                    group_stats = defaultdict(lambda: {'total': 0, 'error': 0, 'error_iters': []})
                    
                    for group_id, iter_num in type_items:
                        group_stats[group_id]['error'] += 1
                        group_stats[group_id]['error_iters'].append(iter_num)
                    
                    # 计算每个组的总迭代数
                    for group_id, iter_nums in sorted(groups.items()):
                        group_stats[group_id]['total'] = len(iter_nums)
                    
                    print(f"      {'组ID':<8} {'总迭代数':<12} {'错误迭代数':<12} {'错误率':<12} {'错误迭代列表'}")
                    output_file.write(f"      {'组ID':<8} {'总迭代数':<12} {'错误迭代数':<12} {'错误率':<12} {'错误迭代列表'}\n")
                    print(f"      {'-'*80}")
                    output_file.write(f"      {'-'*80}\n")
                    
                    for group_id in sorted(group_stats.keys()):
                        stats = group_stats[group_id]
                        stats['error_iters'] = sorted(stats['error_iters'])
                        stats['error_rate'] = stats['error'] / stats['total'] if stats['total'] > 0 else 0
                        error_iters_str = str(stats['error_iters']) if stats['error_iters'] else '[]'
                        print(f"      {group_id:<8} {stats['total']:<12} {stats['error']:<12} {stats['error_rate']:.4f}      {error_iters_str}")
                        output_file.write(f"      {group_id:<8} {stats['total']:<12} {stats['error']:<12} {stats['error_rate']:.4f}      {error_iters_str}\n")
                    
                    # 统计哪些组发生了该类型错误
                    groups_with_this_error = [gid for gid, stats in group_stats.items() if stats['error'] > 0]
                    print(f"\n      发生该类型错误的组: {sorted(groups_with_this_error)} (共 {len(groups_with_this_error)} 个组)")
                    output_file.write(f"\n      发生该类型错误的组: {sorted(groups_with_this_error)} (共 {len(groups_with_this_error)} 个组)\n")
            else:
                # 其他错误类型按常规方式统计
                # 统计每个组中的错误情况
                group_error_stats = {}  # {group_id: {'total': count, 'error': count, 'error_iters': []}}
                
                for group_id, iter_nums in sorted(groups.items()):
                    total_count = len(iter_nums)
                    error_count = 0
                    error_iters = []
                    
                    for iter_num in iter_nums:
                        iter_idx = int(iter_num) - 1  # 迭代号从1开始，索引从0开始
                        if 0 <= iter_idx < len(error_list):
                            if error_list[iter_idx] > 0:
                                error_count += 1
                                error_iters.append(int(iter_num))
                    
                    group_error_stats[group_id] = {
                        'total': total_count,
                        'error': error_count,
                        'error_iters': sorted(error_iters),
                        'error_rate': error_count / total_count if total_count > 0 else 0
                    }
                
                # 打印统计结果
                print(f"\n  错误类型: {error_key}")
                output_file.write(f"\n  错误类型: {error_key}\n")
                print(f"  {'组ID':<8} {'总迭代数':<12} {'错误迭代数':<12} {'错误率':<12} {'错误迭代列表'}")
                output_file.write(f"  {'组ID':<8} {'总迭代数':<12} {'错误迭代数':<12} {'错误率':<12} {'错误迭代列表'}\n")
                print(f"  {'-'*80}")
                output_file.write(f"  {'-'*80}\n")
                
                for group_id in sorted(group_error_stats.keys()):
                    stats = group_error_stats[group_id]
                    error_iters_str = str(stats['error_iters']) if stats['error_iters'] else '[]'
                    print(f"  {group_id:<8} {stats['total']:<12} {stats['error']:<12} {stats['error_rate']:.4f}      {error_iters_str}")
                    output_file.write(f"  {group_id:<8} {stats['total']:<12} {stats['error']:<12} {stats['error_rate']:.4f}      {error_iters_str}\n")
                
                # 统计哪些组发生了错误
                groups_with_errors = [gid for gid, stats in group_error_stats.items() if stats['error'] > 0]
                print(f"\n  发生错误的组: {sorted(groups_with_errors)} (共 {len(groups_with_errors)} 个组)")
                output_file.write(f"\n  发生错误的组: {sorted(groups_with_errors)} (共 {len(groups_with_errors)} 个组)\n")
                
                # 统计错误是否集中在少数特定组中
                total_errors = sum(stats['error'] for stats in group_error_stats.values())
                if total_errors > 0:
                    # 按错误数量排序
                    sorted_groups = sorted(group_error_stats.items(), key=lambda x: x[1]['error'], reverse=True)
                    print(f"\n  错误分布分析:")
                    output_file.write(f"\n  错误分布分析:\n")
                    
                    # 统计前几个组的错误占比
                    for top_n in [1, 2, 3]:
                        if top_n <= len(sorted_groups):
                            top_errors = sum(stats['error'] for _, stats in sorted_groups[:top_n])
                            percentage = (top_errors / total_errors * 100) if total_errors > 0 else 0
                            top_groups = [gid for gid, _ in sorted_groups[:top_n]]
                            print(f"    前 {top_n} 个组 ({top_groups}) 的错误占比: {percentage:.2f}% ({top_errors}/{total_errors})")
                            output_file.write(f"    前 {top_n} 个组 ({top_groups}) 的错误占比: {percentage:.2f}% ({top_errors}/{total_errors})\n")


def main():
    vectors_dir = 'pylot/cov_vector'
    nac_vectors_dir = 'pylot/error_seeds_vectors'
    error_data_file = 'result/datas/100_iter_errors.json'
    output_file_path = 'module_vec_type_num_analys_output9_3.txt'
    
    print("=" * 80)
    print("开始分析模块向量类型和神经元覆盖向量相似性")
    print("=" * 80)
    # print("注意: nac_vector只使用300-400段（索引300到399，共100个元素）进行相似度计算")
    print("=" * 80)
    
    # 读取错误数据
    error_data = None
    if os.path.exists(error_data_file):
        print(f"\n正在读取错误数据文件: {error_data_file}")
        error_data = read_error_data(error_data_file)
        if error_data:
            print("错误数据读取成功")
        else:
            print("警告: 错误数据读取失败，将跳过错误关联分析")
    else:
        print(f"\n警告: 错误数据文件不存在: {error_data_file}")
        print("将跳过错误关联分析")
    
    # 读取所有迭代的模块向量和神经元覆盖向量
    module_vectors = {
        'perception_vector': {},
        'planning_vector': {},
        'control_vector': {},
        'total_vector': {}
    }
    nac_vectors = {}
    
    # 读取模块向量
    if os.path.exists(vectors_dir):
        for filename in sorted(os.listdir(vectors_dir)):
            if filename.endswith('_array_vector.npy'):
                iter_num = filename.replace('_array_vector.npy', '')
                file_path = os.path.join(vectors_dir, filename)
                data = read_npy_file(file_path)
                
                if data and isinstance(data, dict):
                    for module_name in module_vectors.keys():
                        if module_name in data and isinstance(data[module_name], np.ndarray):
                            module_vectors[module_name][iter_num] = data[module_name]
    
    # 读取神经元覆盖向量
    nac_read_count = 0
    nac_failed_count = 0
    
    if os.path.exists(nac_vectors_dir):
        for filename in sorted(os.listdir(nac_vectors_dir)):
            if filename.endswith('_error.pickle') or filename.endswith('_normal.pickle'):
                iter_num = filename.replace('_error.pickle', '').replace('_normal.pickle', '')
                file_path = os.path.join(nac_vectors_dir, filename)
                data = read_pickle_file(file_path)
                
                if data is not None:
                    nac_vector = None
                    
                    # 如果pickle文件本身是一个字典，直接检查
                    if isinstance(data, dict) and 'nac_vector' in data:
                        nac_vector = data['nac_vector']
                    # 如果pickle文件是一个列表，遍历查找包含'nac_vector'的字典
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'nac_vector' in item:
                                nac_vector = item['nac_vector']
                                break
                    
                    # 处理nac_vector（可能是list或ndarray）
                    if nac_vector is not None:
                        # 如果是list，转换为numpy数组
                        if isinstance(nac_vector, list):
                            nac_vector = np.array(nac_vector)
                        # 如果是numpy数组，只取300-400这一段（索引300到399，共100个元素）
                        if isinstance(nac_vector, np.ndarray):
                            original_len = len(nac_vector)
                            # 只取300-400这一段
                            if len(nac_vector) > 400:
                                nac_vector = nac_vector[300:400]
                            elif len(nac_vector) > 300:
                                # 如果向量长度不足400，只取到末尾
                                nac_vector = nac_vector[300:]
                            else:
                                # 如果向量长度不足300，使用空向量
                                nac_vector = np.array([])
                                print(f"警告: 迭代 {iter_num} 的nac_vector长度不足300（实际长度: {original_len}）")
                            nac_vectors[iter_num] = nac_vector
                            nac_read_count += 1
                        else:
                            nac_failed_count += 1
                    else:
                        nac_failed_count += 1
                else:
                    nac_failed_count += 1
    
    print(f"\n神经元覆盖向量读取统计:")
    print(f"  成功读取: {nac_read_count} 个")
    print(f"  读取失败: {nac_failed_count} 个")
    
    if nac_read_count == 0 and nac_failed_count > 0:
        print(f"\n警告: 尝试读取 {nac_failed_count} 个pickle文件，但都没有成功读取到神经元覆盖向量")
        print("请检查pickle文件的数据结构")
        # 尝试读取一个文件来调试
        if os.path.exists(nac_vectors_dir):
            sample_files = [f for f in os.listdir(nac_vectors_dir) 
                          if (f.endswith('_error.pickle') or f.endswith('_normal.pickle'))][:3]
            for sample_file in sample_files:
                file_path = os.path.join(nac_vectors_dir, sample_file)
                data = read_pickle_file(file_path)
                if data is not None:
                    print(f"\n示例文件 {sample_file} 的数据结构:")
                    print(f"  类型: {type(data)}")
                    if isinstance(data, dict):
                        print(f"  键: {list(data.keys())}")
                        if 'nac_vector' in data:
                            print(f"  nac_vector类型: {type(data['nac_vector'])}")
                            print(f"  nac_vector是否为ndarray: {isinstance(data['nac_vector'], np.ndarray)}")
                    elif isinstance(data, list):
                        print(f"  列表长度: {len(data)}")
                        if len(data) > 0:
                            print(f"  第一个元素类型: {type(data[0])}")
                            if isinstance(data[0], dict):
                                print(f"  第一个元素的键: {list(data[0].keys())}")
                    break
    
    print(f"\n读取到 {len(module_vectors['perception_vector'])} 个迭代的模块向量")
    print(f"读取到 {len(nac_vectors)} 个迭代的神经元覆盖向量")
    
    # 打开输出文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("模块向量类型和神经元覆盖向量相似性分析报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"\n读取到 {len(module_vectors['perception_vector'])} 个迭代的模块向量\n")
        f.write(f"读取到 {len(nac_vectors)} 个迭代的神经元覆盖向量\n")
        
        # 分析每个模块
        for module_name in ['perception_vector', 'planning_vector', 'control_vector', 'total_vector']:
            if module_vectors[module_name]:
                analyze_module(module_name, module_vectors[module_name], nac_vectors, f, error_data)
    
    print(f"\n分析完成！结果已保存到: {output_file_path}")


if __name__ == '__main__':
    main()
