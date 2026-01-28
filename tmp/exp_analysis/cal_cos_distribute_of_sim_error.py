

import os
import json
import pickle
import numpy as np
import prettytable

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
    valid_mask = ~((vec1 == -1) & (vec2 == -1))
    
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


def classify_similarity(sim):
    """
    根据相似度值分类到不同的区间
    返回区间索引：0=x=100%, 1=100%>x>=90%, 2=90%>x>=80%, 3=80%>x>=70%, 4=70%>x>=60%, 5=60%>x
    """
    if sim >= 1.0:
        return 0  # x=100%
    elif sim >= 0.9:
        return 1  # 100%>x>=90%
    elif sim >= 0.8:
        return 2  # 90%>x>=80%
    elif sim >= 0.7:
        return 3  # 80%>x>=70%
    elif sim >= 0.6:
        return 4  # 70%>x>=60%
    else:
        return 5  # 60%>x


def classify_max_similarity(max_sim):
    """
    根据最大相似度值分类到不同的区间
    返回区间索引：0=最大值为1, 1=最大值>=0.9, 2=最大值>=0.8, 3=最大值>=0.7, 4=最大值>=0.6, 5=最大值<0.6
    """
    if max_sim >= 1.0:
        return 0  # 最大值为1
    elif max_sim >= 0.9:
        return 1  # 最大值>=0.9
    elif max_sim >= 0.8:
        return 2  # 最大值>=0.8
    elif max_sim >= 0.7:
        return 3  # 最大值>=0.7
    elif max_sim >= 0.6:
        return 4  # 最大值>=0.6
    else:
        return 5  # 最大值<0.6


def classify_min_similarity(min_sim):
    """
    根据最小相似度值分类到不同的区间
    返回区间索引：0=最小值为1, 1=最小值>=0.9, 2=最小值>=0.8, 3=最小值>=0.7, 4=最小值>=0.6, 5=最小值<0.6
    """
    if min_sim >= 1.0:
        return 0  # 最小值为1
    elif min_sim >= 0.9:
        return 1  # 最小值>=0.9
    elif min_sim >= 0.8:
        return 2  # 最小值>=0.8
    elif min_sim >= 0.7:
        return 3  # 最小值>=0.7
    elif min_sim >= 0.6:
        return 4  # 最小值>=0.6
    else:
        return 5  # 最小值<0.6


def main():
    # 文件路径
    json_file = "result/datas/100_iter_errors.json"
    vectors_dir = "pylot/cov_vector"
    pickle_dir = "pylot/error_seeds_vectors"
    
    # 1. 读取JSON文件
    print("=" * 100)
    print("步骤1: 读取错误信息JSON文件")
    print("=" * 100)
    json_data = read_json_file(json_file)
    if json_data is None:
        print("无法读取JSON文件，程序退出")
        return
    
    # 2. 分析错误信息，确定每个迭代是否有对应模块的错误
    print("\n" + "=" * 100)
    print("步骤2: 分析错误信息，确定每个迭代的错误类型")
    print("=" * 100)
    
    error_types = ['cur_code_errors', 'cur_model_errors', 'cur_planning_errors', 'cur_control_errors']
    error_groups = {}
    for error_type in error_types:
        error_groups[error_type] = {
            'has_error': [],
            'no_error': []
        }
    
    # 遍历每个错误类型
    for error_type in error_types:
        if error_type in json_data:
            error_list = json_data[error_type]
            for idx, error_value in enumerate(error_list):
                iteration = idx + 1
                if error_value > 0:
                    error_groups[error_type]['has_error'].append(iteration)
                else:
                    error_groups[error_type]['no_error'].append(iteration)
    
    # 打印统计信息
    for error_type in error_types:
        has_count = len(error_groups[error_type]['has_error'])
        no_count = len(error_groups[error_type]['no_error'])
        print(f"{error_type}: 有错误={has_count}, 无错误={no_count}")
    
    # 3. 读取覆盖向量
    print("\n" + "=" * 100)
    print("步骤3: 读取覆盖向量")
    print("=" * 100)
    
    module_names = ['perception_vector', 'planning_vector', 'control_vector', 'other_vector', 'nac_vector']
    iteration_vectors = {}
    
    # 获取所有需要读取的迭代
    all_iterations = set()
    for error_type in error_types:
        all_iterations.update(error_groups[error_type]['has_error'])
        all_iterations.update(error_groups[error_type]['no_error'])
    
    print(f"需要读取 {len(all_iterations)} 个迭代的覆盖向量")
    
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
                        iteration_vectors[iteration][module_name] = np.array(data[module_name])
        else:
            print(f"警告: 迭代 {iteration} 的向量文件不存在: {vector_file}")
        
        # 读取pickle文件中的nac_vector
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
        
        if nac_vector is not None:
            if isinstance(nac_vector, list):
                nac_vector = np.array(nac_vector)
            iteration_vectors[iteration]['nac_vector'] = nac_vector
    
    print(f"成功读取 {len(iteration_vectors)} 个迭代的覆盖向量")
    
    # 4. 计算相似度并分类
    print("\n" + "=" * 100)
    print("步骤4: 计算相似度并分类")
    print("=" * 100)
    
    # 相似度区间标签（基于平均值）
    similarity_ranges = [
        "x=100%",
        "100%>x>=90%",
        "90%>x>=80%",
        "80%>x>=70%",
        "70%>x>=60%",
        "60%>x"
    ]
    
    # 相似度区间标签（基于最大值）
    max_similarity_ranges = [
        "最大值为1",
        "最大值>=0.9",
        "最大值>=0.8",
        "最大值>=0.7",
        "最大值>=0.6",
        "最大值<0.6"
    ]
    
    # 相似度区间标签（基于最小值）
    min_similarity_ranges = [
        "最小值为1",
        "最小值>=0.9",
        "最小值>=0.8",
        "最小值>=0.7",
        "最小值>=0.6",
        "最小值<0.6"
    ]
    
    # 模块名称映射（用于表格显示）
    module_display_names = {
        'perception_vector': 'Perception',
        'planning_vector': 'planning',
        'control_vector': 'Control',
        'other_vector': 'Other',
        'nac_vector': 'NAC'  # 虽然注释中没有提到NAC，但代码中有，这里也处理
    }
    
    # 错误类型显示名称映射
    error_display_names = {
        'cur_code_errors': 'cur_code_errors',
        'cur_model_errors': 'cur_model_errors',
        'cur_planning_errors': 'cur_planning_errors',
        'cur_control_errors': 'cur_control_errors'
    }
    
    # 存储统计结果（基于平均值）
    # 结构：{module_name: {error_type: [count_per_range]}}
    statistics = {}
    for module_name in module_names:
        statistics[module_name] = {}
        for error_type in error_types:
            statistics[module_name][error_type] = [0] * 6  # 6个区间
    
    # 存储统计结果（基于最大值）
    # 结构：{module_name: {error_type: [count_per_range]}}
    statistics_max = {}
    for module_name in module_names:
        statistics_max[module_name] = {}
        for error_type in error_types:
            statistics_max[module_name][error_type] = [0] * 6  # 6个区间
    
    # 存储统计结果（基于最小值）
    # 结构：{module_name: {error_type: [count_per_range]}}
    statistics_min = {}
    for module_name in module_names:
        statistics_min[module_name] = {}
        for error_type in error_types:
            statistics_min[module_name][error_type] = [0] * 6  # 6个区间
    
    # 对每种错误类型和每个模块进行计算
    for error_type in error_types:
        print(f"\n处理 {error_type}...")
        has_error_iters = error_groups[error_type]['has_error']
        no_error_iters = error_groups[error_type]['no_error']
        
        if not has_error_iters or not no_error_iters:
            print(f"  跳过 {error_type}：有错误迭代={len(has_error_iters)}, 无错误迭代={len(no_error_iters)}")
            continue
        
        # 对每个有错误的迭代，计算其与所有无错误迭代的平均相似度
        for error_iter in has_error_iters:
            if error_iter not in iteration_vectors:
                continue
            
            # 对每个模块计算平均相似度
            for module_name in module_names:
                if module_name not in iteration_vectors[error_iter]:
                    continue
                
                error_vec = iteration_vectors[error_iter][module_name]
                similarities = []
                
                # 计算与所有无错误迭代的相似度
                for no_error_iter in no_error_iters:
                    if no_error_iter not in iteration_vectors:
                        continue
                    if module_name not in iteration_vectors[no_error_iter]:
                        continue
                    
                    no_error_vec = iteration_vectors[no_error_iter][module_name]
                    sim, error_msg = cosine_similarity(error_vec, no_error_vec)
                    if error_msg is None and sim is not None:
                        similarities.append(sim)
                
                # 计算平均相似度并分类
                if similarities:
                    avg_sim = np.mean(similarities)
                    range_idx = classify_similarity(avg_sim)
                    statistics[module_name][error_type][range_idx] += 1
                    
                    # 计算最大相似度并分类
                    max_sim = np.max(similarities)
                    max_range_idx = classify_max_similarity(max_sim)
                    statistics_max[module_name][error_type][max_range_idx] += 1
                    
                    # 计算最小相似度并分类
                    min_sim = np.min(similarities)
                    min_range_idx = classify_min_similarity(min_sim)
                    statistics_min[module_name][error_type][min_range_idx] += 1
    
    # 5. 生成表格（基于平均值）
    print("\n" + "=" * 100)
    print("步骤5: 生成结果表格（基于平均值）")
    print("=" * 100)
    
    # 为每个模块生成表格
    for module_name in ['perception_vector', 'planning_vector', 'control_vector', 'other_vector']:
        if module_name not in statistics:
            continue
        
        module_display = module_display_names.get(module_name, module_name)
        print(f"\n{module_display} 模块:")
        print("-" * 100)
        
        # 创建表格
        table = prettytable.PrettyTable()
        table.field_names = ["错误类型"] + similarity_ranges + ["总计"]
        
        # 添加每种错误类型的行
        for error_type in error_types:
            counts = statistics[module_name][error_type]
            total = sum(counts)
            row = [error_display_names[error_type]] + counts + [total]
            table.add_row(row)
        
        print(table)
    
    # 6. 生成表格（基于最大值）
    print("\n" + "=" * 100)
    print("步骤6: 生成结果表格（基于最大值）")
    print("=" * 100)
    
    # 为每个模块生成表格
    for module_name in ['perception_vector', 'planning_vector', 'control_vector', 'other_vector']:
        if module_name not in statistics_max:
            continue
        
        module_display = module_display_names.get(module_name, module_name)
        print(f"\n{module_display} 模块:")
        print("-" * 100)
        
        # 创建表格
        table = prettytable.PrettyTable()
        table.field_names = ["错误类型"] + max_similarity_ranges + ["总计"]
        
        # 添加每种错误类型的行
        for error_type in error_types:
            counts = statistics_max[module_name][error_type]
            total = sum(counts)
            row = [error_display_names[error_type]] + counts + [total]
            table.add_row(row)
        
        print(table)
    
    # 7. 生成表格（基于最小值）
    print("\n" + "=" * 100)
    print("步骤7: 生成结果表格（基于最小值）")
    print("=" * 100)
    
    # 为每个模块生成表格
    for module_name in ['perception_vector', 'planning_vector', 'control_vector', 'other_vector']:
        if module_name not in statistics_min:
            continue
        
        module_display = module_display_names.get(module_name, module_name)
        print(f"\n{module_display} 模块:")
        print("-" * 100)
        
        # 创建表格
        table = prettytable.PrettyTable()
        table.field_names = ["错误类型"] + min_similarity_ranges + ["总计"]
        
        # 添加每种错误类型的行
        for error_type in error_types:
            counts = statistics_min[module_name][error_type]
            total = sum(counts)
            row = [error_display_names[error_type]] + counts + [total]
            table.add_row(row)
        
        print(table)
    
    # 打印汇总信息
    print("\n" + "=" * 100)
    print("汇总信息（基于平均值）")
    print("=" * 100)
    for module_name in ['perception_vector', 'planning_vector', 'control_vector', 'other_vector']:
        if module_name not in statistics:
            continue
        module_display = module_display_names.get(module_name, module_name)
        print(f"\n{module_display}:")
        for error_type in error_types:
            counts = statistics[module_name][error_type]
            total = sum(counts)
            print(f"  {error_display_names[error_type]}: 总计={total}")
    
    print("\n" + "=" * 100)
    print("汇总信息（基于最大值）")
    print("=" * 100)
    for module_name in ['perception_vector', 'planning_vector', 'control_vector', 'other_vector']:
        if module_name not in statistics_max:
            continue
        module_display = module_display_names.get(module_name, module_name)
        print(f"\n{module_display}:")
        for error_type in error_types:
            counts = statistics_max[module_name][error_type]
            total = sum(counts)
            print(f"  {error_display_names[error_type]}: 总计={total}")
    
    print("\n" + "=" * 100)
    print("汇总信息（基于最小值）")
    print("=" * 100)
    for module_name in ['perception_vector', 'planning_vector', 'control_vector', 'other_vector']:
        if module_name not in statistics_min:
            continue
        module_display = module_display_names.get(module_name, module_name)
        print(f"\n{module_display}:")
        for error_type in error_types:
            counts = statistics_min[module_name][error_type]
            total = sum(counts)
            print(f"  {error_display_names[error_type]}: 总计={total}")


if __name__ == "__main__":
    main()
