import os
import json
import pickle
import numpy as np

'''
计算触发不同错误的迭代之间不同模块的覆盖向量的余弦相似度。

1.按错误类型划分计算相似度的对象。
2.取出每次迭代的三个模块的覆盖向量perception_vector，planning_vector，control_vector，other_vector。

3.计算"发生cur_code_errors的迭代"和"没有发生cur_code_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度。
计算"发生cur_model_errors的迭代"和"没有发生cur_model_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度。
计算"发生cur_planning_errors的迭代"和"没有发生cur_planning_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度。
计算"发生cur_control_errors的迭代"和"没有发生cur_control_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度。

计算触发相同错误的迭代之间不同模块的覆盖向量的余弦相似度。
4.计算所有"发生cur_code_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度。
计算所有"发生cur_model_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度。
计算所有"发生cur_planning_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度。
计算所有"发生cur_control_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度。

计算没有触发某错误的迭代之间不同模块的覆盖向量的余弦相似度。
5.计算所有"没有发生cur_code_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度。
计算所有"没有发生cur_model_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度。
计算所有"没有发生cur_planning_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度。
计算所有"没有发生cur_control_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度。


统计各个模块中“发生某错误类型中迭代”和“没有发生某错误类型中迭代”之间的覆盖向量的余弦相似度的分布情况。
如等于1的相似度数量，小于1并且大于等于0.9的相似度数量，小于0.9并且大于等于0.8的相似度数量，小于0.8并且大于等于0.7的相似度数量，小于0.7并且大于等于0.6的相似度数量，小于0.6的相似度数量。
6.1"发生cur_code_errors的迭代"和"没有发生cur_code_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度的分布情况。。
6.2"发生cur_model_errors的迭代"和"没有发生cur_model_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度的分布情况。
6.3"发生cur_planning_errors的迭代"和"没有发生cur_planning_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度的分布情况。
6.4"发生cur_control_errors的迭代"和"没有发生cur_control_errors的迭代"之间的覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度的分布情况。


'''


def read_json_file(file_path):
    """读取JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败 {file_path}: {e}")
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


def read_pickle_file(file_path):
    """读取pickle文件并返回字典"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"读取pickle文件失败 {file_path}: {e}")
        return None


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    过滤掉值为-1的位置（-1代表空行和注释，对覆盖向量没有意义）
    """
    # 检查向量长度是否一致
    if len(vec1) != len(vec2):
        return None, f"向量长度不一致: {len(vec1)} vs {len(vec2)}"
    
    # 过滤掉值为-1的位置（只有当vec1和vec2都为-1时，才过滤掉）
    # 使用 ~ 对数组进行按位取反，而不是 not（not只适用于标量）
    valid_mask = ~((vec1 == -1) & (vec2 == -1))
    # valid_mask = ((vec1 == -1) | (vec1 == 1) | (vec1 == 0))

    # valid_mask = ~((vec1 == -1) & (vec2 == -1)) & ~((vec1 == 1) & (vec2 == 1))
    
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


def get_average_vector(vectors):
    """
    计算多个向量的平均向量
    如果向量长度不一致，以最长向量为准，短向量用0填充
    """
    if not vectors:
        return None
    
    # 找到最大长度
    max_len = max(len(v) for v in vectors)
    
    # 将所有向量填充到相同长度
    padded_vectors = []
    for vec in vectors:
        if len(vec) < max_len:
            padding = np.zeros(max_len - len(vec), dtype=vec.dtype)
            padded_vec = np.concatenate([vec, padding])
        else:
            padded_vec = vec
        padded_vectors.append(padded_vec)
    
    # 计算平均向量
    avg_vector = np.mean(padded_vectors, axis=0)
    return avg_vector


def calculate_between_groups_similarity(group1_vectors, group2_vectors):
    """
    计算两组向量之间的两两余弦相似度
    返回平均相似度、最大相似度、最小相似度、中位数、相似度列表
    """
    if not group1_vectors or not group2_vectors:
        return None, None, None, None, [], "至少一组向量为空，无法计算"
    
    similarities = []
    
    # 计算两组向量之间的所有两两组合的余弦相似度
    for vec1 in group1_vectors:
        for vec2 in group2_vectors:
            sim, error_msg = cosine_similarity(vec1, vec2)
            if error_msg is None and sim is not None:
                similarities.append(sim)
    
    if not similarities:
        return None, None, None, None, [], "无法计算任何相似度"
    
    avg_sim = np.mean(similarities)
    max_sim = np.max(similarities)
    min_sim = np.min(similarities)
    median_sim = np.median(similarities)
    
    return avg_sim, max_sim, min_sim, median_sim, similarities, None


def calculate_within_group_similarity(vectors):
    """
    计算同一组内所有向量之间的两两余弦相似度
    返回平均相似度、最大相似度、最小相似度、中位数、相似度列表
    """
    if len(vectors) < 2:
        return None, None, None, None, [], "向量数量少于2个，无法计算"
    
    similarities = []
    
    # 计算所有两两组合的余弦相似度
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim, error_msg = cosine_similarity(vectors[i], vectors[j])
            if error_msg is None and sim is not None:
                similarities.append(sim)
    
    if not similarities:
        return None, None, None, None, [], "无法计算任何相似度"
    
    avg_sim = np.mean(similarities)
    max_sim = np.max(similarities)
    min_sim = np.min(similarities)
    median_sim = np.median(similarities)
    
    return avg_sim, max_sim, min_sim, median_sim, similarities, None


def calculate_similarity_distribution(similarities):
    """
    统计相似度的分布情况
    返回各个区间的数量
    """
    if not similarities:
        return None
    
    distribution = {
        'x=1.0': 0,           # 等于1
        '0.9<=x<1.0': 0,      # 小于1并且大于等于0.9
        '0.8<=x<0.9': 0,      # 小于0.9并且大于等于0.8
        '0.7<=x<0.8': 0,      # 小于0.8并且大于等于0.7
        '0.6<=x<0.7': 0,      # 小于0.7并且大于等于0.6
        'x<0.6': 0            # 小于0.6
    }
    
    for sim in similarities:
        if sim == 1.0:
            distribution['x=1.0'] += 1
        elif 0.9 <= sim < 1.0:
            distribution['0.9<=x<1.0'] += 1
        elif 0.8 <= sim < 0.9:
            distribution['0.8<=x<0.9'] += 1
        elif 0.7 <= sim < 0.8:
            distribution['0.7<=x<0.8'] += 1
        elif 0.6 <= sim < 0.7:
            distribution['0.6<=x<0.7'] += 1
        else:
            distribution['x<0.6'] += 1
    
    return distribution


def main():
    # 文件路径
    json_file = "result/datas/100_iter_errors.json"
    vectors_dir = "pylot/cov_vector"
    
    # 1. 读取JSON文件
    print("=" * 100)
    print("步骤1: 读取错误信息JSON文件")
    print("=" * 100)
    json_data = read_json_file(json_file)
    if json_data is None:
        print("无法读取JSON文件，程序退出")
        return
    
    print(f"成功读取JSON文件，包含 {len(json_data)} 个键")
    # 检查错误类型列表的长度
    if 'cur_code_errors' in json_data:
        print(f"每个错误类型列表包含 {len(json_data['cur_code_errors'])} 个迭代的数据")
    
    # 2. 分析错误信息，确定每个迭代是否有对应模块的错误
    print("\n" + "=" * 100)
    print("步骤2: 分析错误信息，确定每个迭代的错误类型")
    print("=" * 100)
    
    # 错误类型列表
    error_types = ['cur_code_errors', 'cur_model_errors', 'cur_planning_errors', 'cur_control_errors']
    
    # 存储每个错误类型的迭代分组
    error_groups = {}
    for error_type in error_types:
        error_groups[error_type] = {
            'has_error': [],  # 有错误的迭代
            'no_error': []    # 无错误的迭代
        }
    
    # JSON文件结构：顶层是字典，每个错误类型对应一个列表，列表索引+1就是迭代号
    for error_type in error_types:
        if error_type in json_data:
            error_list = json_data[error_type]
            # 遍历列表，索引+1就是迭代号
            for idx, error_value in enumerate(error_list):
                iteration = idx + 1  # 迭代号从1开始
                # 如果错误值大于0，则认为有错误
                if error_value > 0:
                    error_groups[error_type]['has_error'].append(iteration)
                else:
                    error_groups[error_type]['no_error'].append(iteration)
        else:
            print(f"警告: JSON文件中没有找到 {error_type}")
    
    # 打印统计信息
    for error_type in error_types:
        has_count = len(error_groups[error_type]['has_error'])
        no_count = len(error_groups[error_type]['no_error'])
        print(f"{error_type}:")
        print(f"  有错误: {has_count} 个迭代")
        print(f"  无错误: {no_count} 个迭代")
    
    # 3. 读取覆盖向量
    print("\n" + "=" * 100)
    print("步骤3: 读取覆盖向量")
    print("=" * 100)
    
    # 模块向量名称（包括nac_vector）
    module_names = ['perception_vector', 'planning_vector', 'control_vector', 'other_vector', 'nac_vector']
    
    # 存储每个迭代的模块向量
    iteration_vectors = {}  # {iteration: {module_name: vector}}
    
    # 获取所有需要读取的迭代
    all_iterations = set()
    for error_type in error_types:
        all_iterations.update(error_groups[error_type]['has_error'])
        all_iterations.update(error_groups[error_type]['no_error'])
    
    print(f"需要读取 {len(all_iterations)} 个迭代的覆盖向量")
    
    # pickle文件路径
    pickle_dir = "/media/lzq/D/lzq/pylot_test/pylot/error_seeds_vectors"
    
    # 读取每个迭代的向量
    for iteration in sorted(all_iterations):
        iteration_vectors[iteration] = {}
        
        # 读取npy文件中的模块向量
        vector_file = os.path.join(vectors_dir, f"{iteration}_array_vector.npy")
        if os.path.exists(vector_file):
            data = read_npy_file(vector_file)
            if data and isinstance(data, dict):
                for module_name in ['perception_vector', 'planning_vector', 'control_vector', 'other_vector']:
                    if module_name in data:
                        iteration_vectors[iteration][module_name] = data[module_name]
                    else:
                        print(f"警告: 迭代 {iteration} 缺少模块向量 {module_name}")
        else:
            print(f"警告: 迭代 {iteration} 的向量文件不存在: {vector_file}")
        
        # 读取pickle文件中的nac_vector
        # 尝试读取_error.pickle或_normal.pickle
        error_pickle_file = os.path.join(pickle_dir, f"{iteration}_error.pickle")
        normal_pickle_file = os.path.join(pickle_dir, f"{iteration}_normal.pickle")
        
        nac_vector = None
        if os.path.exists(error_pickle_file):
            pickle_data = read_pickle_file(error_pickle_file)
            if pickle_data and isinstance(pickle_data, dict) and 'nac_vector' in pickle_data:
                nac_vector = pickle_data['nac_vector']
        elif os.path.exists(normal_pickle_file):
            pickle_data = read_pickle_file(normal_pickle_file)
            if pickle_data and isinstance(pickle_data, dict) and 'nac_vector' in pickle_data:
                nac_vector = pickle_data['nac_vector']
        else:
            print(f"警告: 迭代 {iteration} 的pickle文件不存在（既没有_error.pickle也没有_normal.pickle）")
        
        if nac_vector is not None:
            # 确保nac_vector是numpy数组
            if isinstance(nac_vector, list):
                nac_vector = np.array(nac_vector)
            # 只取429-858这一段（索引429到857，共430个元素）
            # 546-567
            original_len = len(nac_vector) if isinstance(nac_vector, np.ndarray) else 0
            if isinstance(nac_vector, np.ndarray):
                if len(nac_vector) > 567:
                    nac_vector = nac_vector[546:567]
                elif len(nac_vector) > 546:
                    # 如果向量长度不足858，只取到末尾
                    nac_vector = nac_vector[546:]
                else:
                    # 如果向量长度不足429，使用空向量
                    nac_vector = np.array([])
                    print(f"警告: 迭代 {iteration} 的nac_vector长度不足546（实际长度: {original_len}）")
            iteration_vectors[iteration]['nac_vector'] = nac_vector
        else:
            print(f"警告: 迭代 {iteration} 缺少nac_vector")
    
    print(f"成功读取 {len(iteration_vectors)} 个迭代的覆盖向量")
    
    # 4. 计算余弦相似度
    print("\n" + "=" * 100)
    print("步骤4: 计算余弦相似度")
    print("=" * 100)
    
    results = {}
    
    for error_type in error_types:
        print(f"\n{error_type}:")
        print("-" * 100)
        
        has_error_iters = error_groups[error_type]['has_error']
        no_error_iters = error_groups[error_type]['no_error']
        
        results[error_type] = {}
        
        for module_name in module_names:
            # 收集有错误组的向量
            has_error_vectors = []
            for iter_num in has_error_iters:
                if iter_num in iteration_vectors and module_name in iteration_vectors[iter_num]:
                    has_error_vectors.append(iteration_vectors[iter_num][module_name])
            
            # 收集无错误组的向量
            no_error_vectors = []
            for iter_num in no_error_iters:
                if iter_num in iteration_vectors and module_name in iteration_vectors[iter_num]:
                    no_error_vectors.append(iteration_vectors[iter_num][module_name])
            
            if not has_error_vectors or not no_error_vectors:
                print(f"  {module_name}: 无法计算（有错误组: {len(has_error_vectors)} 个向量，无错误组: {len(no_error_vectors)} 个向量）")
                results[error_type][module_name] = None
                continue
            
            # 计算两组向量之间的两两余弦相似度
            avg_sim, max_sim, min_sim, median_sim, sim_list, error_msg = calculate_between_groups_similarity(
                has_error_vectors, no_error_vectors
            )
            
            if error_msg:
                print(f"  {module_name}: 计算失败 - {error_msg}")
                results[error_type][module_name] = None
            else:
                print(f"  {module_name}:")
                print(f"    有错误组向量数: {len(has_error_vectors)}")
                print(f"    无错误组向量数: {len(no_error_vectors)}")
                print(f"    两两组合数: {len(sim_list)}")
                print(f"    平均相似度: {avg_sim:.6f}")
                print(f"    最大相似度: {max_sim:.6f}")
                print(f"    最小相似度: {min_sim:.6f}")
                print(f"    中位数相似度: {median_sim:.6f}")
                results[error_type][module_name] = {
                    'avg': avg_sim,
                    'max': max_sim,
                    'min': min_sim,
                    'median': median_sim,
                    'count': len(sim_list),
                    'similarities': sim_list  # 保存相似度列表用于分布统计
                }
    
    # 5. 计算相同错误组内的余弦相似度
    print("\n" + "=" * 100)
    print("步骤5: 计算相同错误组内的余弦相似度")
    print("=" * 100)
    
    within_group_results = {}
    
    for error_type in error_types:
        print(f"\n{error_type} (相同错误组内):")
        print("-" * 100)
        
        has_error_iters = error_groups[error_type]['has_error']
        
        if len(has_error_iters) < 2:
            print(f"  有错误的迭代数量少于2个（{len(has_error_iters)}），无法计算组内相似度")
            within_group_results[error_type] = {}
            continue
        
        within_group_results[error_type] = {}
        
        for module_name in module_names:
            # 收集有错误组的向量
            has_error_vectors = []
            for iter_num in has_error_iters:
                if iter_num in iteration_vectors and module_name in iteration_vectors[iter_num]:
                    has_error_vectors.append(iteration_vectors[iter_num][module_name])
            
            if len(has_error_vectors) < 2:
                print(f"  {module_name}: 无法计算（有效向量数: {len(has_error_vectors)}，需要至少2个）")
                within_group_results[error_type][module_name] = None
                continue
            
            # 计算组内相似度
            avg_sim, max_sim, min_sim, median_sim, sim_list, error_msg = calculate_within_group_similarity(has_error_vectors)
            
            if error_msg:
                print(f"  {module_name}: 计算失败 - {error_msg}")
                within_group_results[error_type][module_name] = None
            else:
                print(f"  {module_name}:")
                print(f"    向量数量: {len(has_error_vectors)}")
                print(f"    两两组合数: {len(sim_list)}")
                print(f"    平均相似度: {avg_sim:.6f}")
                print(f"    最大相似度: {max_sim:.6f}")
                print(f"    最小相似度: {min_sim:.6f}")
                print(f"    中位数相似度: {median_sim:.6f}")
                within_group_results[error_type][module_name] = {
                    'avg': avg_sim,
                    'max': max_sim,
                    'min': min_sim,
                    'median': median_sim,
                    'count': len(sim_list)
                }
    
    # 6. 计算没有发生错误的迭代之间的余弦相似度
    print("\n" + "=" * 100)
    print("步骤6: 计算没有发生错误的迭代之间的余弦相似度")
    print("=" * 100)
    
    no_error_group_results = {}
    
    for error_type in error_types:
        print(f"\n{error_type} (无错误组内):")
        print("-" * 100)
        
        no_error_iters = error_groups[error_type]['no_error']
        
        if len(no_error_iters) < 2:
            print(f"  无错误的迭代数量少于2个（{len(no_error_iters)}），无法计算组内相似度")
            no_error_group_results[error_type] = {}
            continue
        
        no_error_group_results[error_type] = {}
        
        for module_name in module_names:
            # 收集无错误组的向量
            no_error_vectors = []
            for iter_num in no_error_iters:
                if iter_num in iteration_vectors and module_name in iteration_vectors[iter_num]:
                    no_error_vectors.append(iteration_vectors[iter_num][module_name])
            
            if len(no_error_vectors) < 2:
                print(f"  {module_name}: 无法计算（有效向量数: {len(no_error_vectors)}，需要至少2个）")
                no_error_group_results[error_type][module_name] = None
                continue
            
            # 计算组内相似度
            avg_sim, max_sim, min_sim, median_sim, sim_list, error_msg = calculate_within_group_similarity(no_error_vectors)
            
            if error_msg:
                print(f"  {module_name}: 计算失败 - {error_msg}")
                no_error_group_results[error_type][module_name] = None
            else:
                print(f"  {module_name}:")
                print(f"    向量数量: {len(no_error_vectors)}")
                print(f"    两两组合数: {len(sim_list)}")
                print(f"    平均相似度: {avg_sim:.6f}")
                print(f"    最大相似度: {max_sim:.6f}")
                print(f"    最小相似度: {min_sim:.6f}")
                print(f"    中位数相似度: {median_sim:.6f}")
                no_error_group_results[error_type][module_name] = {
                    'avg': avg_sim,
                    'max': max_sim,
                    'min': min_sim,
                    'median': median_sim,
                    'count': len(sim_list)
                }
    
    # 7. 打印汇总结果
    print("\n" + "=" * 100)
    print("步骤7: 汇总结果")
    print("=" * 100)
    
    print("\n【不同组之间的余弦相似度 - 平均值】")
    print("-" * 100)
    print(f"{'错误类型':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    
    for error_type in error_types:
        row = f"{error_type:<25}"
        for module_name in module_names:
            result = results[error_type].get(module_name)
            if result is not None and isinstance(result, dict) and 'avg' in result:
                row += f"{result['avg']:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("\n【不同组之间的余弦相似度 - 最大值】")
    print("-" * 100)
    print(f"{'错误类型':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    
    for error_type in error_types:
        row = f"{error_type:<25}"
        for module_name in module_names:
            result = results[error_type].get(module_name)
            if result is not None and isinstance(result, dict) and 'max' in result:
                row += f"{result['max']:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("\n【不同组之间的余弦相似度 - 最小值】")
    print("-" * 100)
    print(f"{'错误类型':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    
    for error_type in error_types:
        row = f"{error_type:<25}"
        for module_name in module_names:
            result = results[error_type].get(module_name)
            if result is not None and isinstance(result, dict) and 'min' in result:
                row += f"{result['min']:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("\n【不同组之间的余弦相似度 - 中位数】")
    print("-" * 100)
    print(f"{'错误类型':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    
    for error_type in error_types:
        row = f"{error_type:<25}"
        for module_name in module_names:
            result = results[error_type].get(module_name)
            if result is not None and isinstance(result, dict) and 'median' in result:
                row += f"{result['median']:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("\n【相同错误组内的余弦相似度 - 平均值】")
    print("-" * 100)
    print(f"{'错误类型':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    
    for error_type in error_types:
        row = f"{error_type:<25}"
        for module_name in module_names:
            result = within_group_results[error_type].get(module_name)
            if result is not None and isinstance(result, dict) and 'avg' in result:
                row += f"{result['avg']:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("\n【相同错误组内的余弦相似度 - 最大值】")
    print("-" * 100)
    print(f"{'错误类型':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    
    for error_type in error_types:
        row = f"{error_type:<25}"
        for module_name in module_names:
            result = within_group_results[error_type].get(module_name)
            if result is not None and isinstance(result, dict) and 'max' in result:
                row += f"{result['max']:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("\n【相同错误组内的余弦相似度 - 最小值】")
    print("-" * 100)
    print(f"{'错误类型':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    
    for error_type in error_types:
        row = f"{error_type:<25}"
        for module_name in module_names:
            result = within_group_results[error_type].get(module_name)
            if result is not None and isinstance(result, dict) and 'min' in result:
                row += f"{result['min']:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("\n【相同错误组内的余弦相似度 - 中位数】")
    print("-" * 100)
    print(f"{'错误类型':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    
    for error_type in error_types:
        row = f"{error_type:<25}"
        for module_name in module_names:
            result = within_group_results[error_type].get(module_name)
            if result is not None and isinstance(result, dict) and 'median' in result:
                row += f"{result['median']:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("\n【无错误组内的余弦相似度 - 平均值】")
    print("-" * 100)
    print(f"{'错误类型':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    
    for error_type in error_types:
        row = f"{error_type:<25}"
        for module_name in module_names:
            result = no_error_group_results[error_type].get(module_name)
            if result is not None and isinstance(result, dict) and 'avg' in result:
                row += f"{result['avg']:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("\n【无错误组内的余弦相似度 - 最大值】")
    print("-" * 100)
    print(f"{'错误类型':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    
    for error_type in error_types:
        row = f"{error_type:<25}"
        for module_name in module_names:
            result = no_error_group_results[error_type].get(module_name)
            if result is not None and isinstance(result, dict) and 'max' in result:
                row += f"{result['max']:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("\n【无错误组内的余弦相似度 - 最小值】")
    print("-" * 100)
    print(f"{'错误类型':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    
    for error_type in error_types:
        row = f"{error_type:<25}"
        for module_name in module_names:
            result = no_error_group_results[error_type].get(module_name)
            if result is not None and isinstance(result, dict) and 'min' in result:
                row += f"{result['min']:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("\n【无错误组内的余弦相似度 - 中位数】")
    print("-" * 100)
    print(f"{'错误类型':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    
    for error_type in error_types:
        row = f"{error_type:<25}"
        for module_name in module_names:
            result = no_error_group_results[error_type].get(module_name)
            if result is not None and isinstance(result, dict) and 'median' in result:
                row += f"{result['median']:<15.6f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("=" * 100)
    
    # 8. 统计相似度分布情况
    print("\n" + "=" * 100)
    print("步骤8: 统计相似度分布情况")
    print("=" * 100)
    print("统计各个模块中'发生某错误类型中迭代'和'没有发生某错误类型中迭代'之间的覆盖向量的余弦相似度分布")
    print("=" * 100)
    
    for error_type in error_types:
        print(f"\n{error_type}:")
        print("-" * 100)
        
        for module_name in module_names:
            result = results[error_type].get(module_name)
            if result is None or not isinstance(result, dict) or 'similarities' not in result:
                print(f"  {module_name}: 无数据")
                continue
            
            similarities = result['similarities']
            distribution = calculate_similarity_distribution(similarities)
            
            if distribution is None:
                print(f"  {module_name}: 无法计算分布")
                continue
            
            print(f"  {module_name}:")
            print(f"    总相似度数量: {len(similarities)}")
            print(f"    等于1.0的数量: {distribution['x=1.0']}")
            print(f"    0.9<=x<1.0的数量: {distribution['0.9<=x<1.0']}")
            print(f"    0.8<=x<0.9的数量: {distribution['0.8<=x<0.9']}")
            print(f"    0.7<=x<0.8的数量: {distribution['0.7<=x<0.8']}")
            print(f"    0.6<=x<0.7的数量: {distribution['0.6<=x<0.7']}")
            print(f"    x<0.6的数量: {distribution['x<0.6']}")
            
            # 计算百分比
            total = len(similarities)
            if total > 0:
                print(f"    等于1.0的比例: {distribution['x=1.0']/total*100:.2f}%")
                print(f"    0.9<=x<1.0的比例: {distribution['0.9<=x<1.0']/total*100:.2f}%")
                print(f"    0.8<=x<0.9的比例: {distribution['0.8<=x<0.9']/total*100:.2f}%")
                print(f"    0.7<=x<0.8的比例: {distribution['0.7<=x<0.8']/total*100:.2f}%")
                print(f"    0.6<=x<0.7的比例: {distribution['0.6<=x<0.7']/total*100:.2f}%")
                print(f"    x<0.6的比例: {distribution['x<0.6']/total*100:.2f}%")
    
    # 9. 以表格形式输出分布情况
    print("\n" + "=" * 100)
    print("步骤9: 相似度分布汇总表格（数量）")
    print("=" * 100)
    
    # 为每个错误类型和模块创建分布表格
    for error_type in error_types:
        print(f"\n【{error_type} - 相似度分布统计（数量）】")
        print("-" * 100)
        print(f"{'模块':<20} {'x=1.0':<12} {'0.9<=x<1.0':<15} {'0.8<=x<0.9':<15} {'0.7<=x<0.8':<15} {'0.6<=x<0.7':<15} {'x<0.6':<12}")
        print("-" * 100)
        
        for module_name in module_names:
            result = results[error_type].get(module_name)
            if result is None or not isinstance(result, dict) or 'similarities' not in result:
                row = f"{module_name:<20} {'N/A':<12} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<12}"
                print(row)
                continue
            
            similarities = result['similarities']
            distribution = calculate_similarity_distribution(similarities)
            
            if distribution is None:
                row = f"{module_name:<20} {'N/A':<12} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<12}"
                print(row)
                continue
            
            row = f"{module_name:<20} "
            row += f"{distribution['x=1.0']:<12} "
            row += f"{distribution['0.9<=x<1.0']:<15} "
            row += f"{distribution['0.8<=x<0.9']:<15} "
            row += f"{distribution['0.7<=x<0.8']:<15} "
            row += f"{distribution['0.6<=x<0.7']:<15} "
            row += f"{distribution['x<0.6']:<12}"
            print(row)
    
    # 10. 按百分比输出分布情况
    print("\n" + "=" * 100)
    print("步骤10: 相似度分布汇总表格（百分比）")
    print("=" * 100)
    
    for error_type in error_types:
        print(f"\n【{error_type} - 相似度分布统计（百分比）】")
        print("-" * 100)
        print(f"{'模块':<20} {'x=1.0':<12} {'0.9<=x<1.0':<15} {'0.8<=x<0.9':<15} {'0.7<=x<0.8':<15} {'0.6<=x<0.7':<15} {'x<0.6':<12}")
        print("-" * 100)
        
        for module_name in module_names:
            result = results[error_type].get(module_name)
            if result is None or not isinstance(result, dict) or 'similarities' not in result:
                row = f"{module_name:<20} {'N/A':<12} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<12}"
                print(row)
                continue
            
            similarities = result['similarities']
            distribution = calculate_similarity_distribution(similarities)
            
            if distribution is None:
                row = f"{module_name:<20} {'N/A':<12} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<12}"
                print(row)
                continue
            
            total = len(similarities)
            if total == 0:
                row = f"{module_name:<20} {'N/A':<12} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<12}"
                print(row)
                continue
            
            row = f"{module_name:<20} "
            row += f"{distribution['x=1.0']/total*100:<11.2f}% "
            row += f"{distribution['0.9<=x<1.0']/total*100:<14.2f}% "
            row += f"{distribution['0.8<=x<0.9']/total*100:<14.2f}% "
            row += f"{distribution['0.7<=x<0.8']/total*100:<14.2f}% "
            row += f"{distribution['0.6<=x<0.7']/total*100:<14.2f}% "
            row += f"{distribution['x<0.6']/total*100:<11.2f}%"
            print(row)
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
