import os
import pickle
import numpy as np

'''
计算原始种子神经元覆盖向量和变异种子代码、神经元覆盖向量的余弦相似度。
原始种子代码覆盖向量路径：/media/lzq/D/lzq/pylot_test/pylot/ori_cov_vector/{iteration}_array_vector.npy
变异种子代码覆盖向量路径：/media/lzq/D/lzq/pylot_test/pylot/cov_vector/{iteration}_array_vector.npy
iteration范围：1-100
每个npy文件中包含若干个覆盖向量，取出其中的perception_vector，planning_vector，control_vector，other_vector，分别代表感知模块、规划模块、控制模块、其他模块的覆盖向量。

原始种子神经元覆盖向量路径：/media/lzq/D/lzq/pylot_test/pylot/ori_seeds_vectors/{i-1}.pickle。i代表迭代号，范围为1-100。取出每次迭代i对应pickle文件中的nac_vector，参与相似度计算。
变异种子神经元覆盖向量路径：/media/lzq/D/lzq/pylot_test/pylot/error_seeds_vectors/{i}_error.pickle（error代表出现代码报错）或者/media/lzq/D/lzq/pylot_test/pylot/error_seeds_vectors/{i}_normal.pickle（normal代表无代码报错），每次迭代i只对应一个pickle文件。i代表迭代号，范围为1-100。取出每次迭代i对应pickle文件中的nac_vector，参与相似度计算。


1.计算原始种子内部各模块覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度，两两迭代之间计算得到平均值、最小值、最大值、中位数。
2.计算变异种子内部各模块覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度，两两迭代之间计算得到平均值、最小值、最大值、中位数。
3.计算原始种子和变异种子覆盖向量之间各模块覆盖向量perception_vector，planning_vector，control_vector，other_vector，nac_vector的余弦相似度，两两迭代之间计算得到平均值、最小值、最大值、中位数。


另外，我需要统计原始种子和变异种子中，代码覆盖向量total_vector和神经元覆盖向量nac_vector的种类数量。即一模一样的向量算作一类，统计总的种类数量。
其中，代码覆盖向量total_vector和各模块覆盖向量perception_vector，planning_vector，control_vector，other_vector一起保存在对应的{i}_array_vector.npy文件中。
给出结果表，列名为（total_code_pattern，nac_pattern），行名为（原始种子，变异生成），值为种类数量。
'''


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


def analyze_vector_similarity(vec1, vec2, segment_size=100, vec1_name="vec1", vec2_name="vec2"):
    """
    分析两个向量的相似性和差异性
    将向量分段，计算每段的相似度，找出完全相同的段和差异明显的段
    
    参数:
        vec1: 第一个向量
        vec2: 第二个向量
        segment_size: 每段的大小（默认100）
        vec1_name: 第一个向量的名称（用于输出）
        vec2_name: 第二个向量的名称（用于输出）
    
    返回:
        analysis_result: 包含分析结果的字典
    """
    # 确保两个向量长度一致
    min_len = min(len(vec1), len(vec2))
    vec1 = vec1[:min_len]
    vec2 = vec2[:min_len]
    
    # 过滤掉值为-1的位置（只有当vec1和vec2都为-1时，才过滤掉）
    valid_mask = ~((vec1 == -1) & (vec2 == -1))
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return {
            'total_length': min_len,
            'valid_length': 0,
            'identical_segments': [],
            'different_segments': [],
            'segment_analysis': []
        }
    
    # 分段分析
    num_segments = (min_len + segment_size - 1) // segment_size
    segment_analysis = []
    identical_segments = []
    different_segments = []
    
    for seg_idx in range(num_segments):
        start_idx = seg_idx * segment_size
        end_idx = min((seg_idx + 1) * segment_size, min_len)
        
        seg_vec1 = vec1[start_idx:end_idx]
        seg_vec2 = vec2[start_idx:end_idx]
        
        # 计算该段的有效位置
        seg_valid_mask = ~((seg_vec1 == -1) & (seg_vec2 == -1))
        seg_valid_vec1 = seg_vec1[seg_valid_mask]
        seg_valid_vec2 = seg_vec2[seg_valid_mask]
        
        if len(seg_valid_vec1) == 0:
            segment_analysis.append({
                'segment': seg_idx,
                'start': start_idx,
                'end': end_idx,
                'length': end_idx - start_idx,
                'valid_length': 0,
                'similarity': None,
                'identical_count': 0,
                'different_count': 0,
                'status': 'all_invalid'
            })
            continue
        
        # 计算该段的相似度
        if len(seg_valid_vec1) == 1:
            seg_sim = 1.0 if seg_valid_vec1[0] == seg_valid_vec2[0] else 0.0
        else:
            dot_product = np.dot(seg_valid_vec1, seg_valid_vec2)
            norm1 = np.linalg.norm(seg_valid_vec1)
            norm2 = np.linalg.norm(seg_valid_vec2)
            if norm1 == 0 or norm2 == 0:
                seg_sim = 0.0
            else:
                seg_sim = dot_product / (norm1 * norm2)
        
        # 统计完全相同的元素数量
        identical_count = np.sum(seg_valid_vec1 == seg_valid_vec2)
        different_count = len(seg_valid_vec1) - identical_count
        
        segment_info = {
            'segment': seg_idx,
            'start': start_idx,
            'end': end_idx,
            'length': end_idx - start_idx,
            'valid_length': len(seg_valid_vec1),
            'similarity': seg_sim,
            'identical_count': identical_count,
            'different_count': different_count,
            'identical_ratio': identical_count / len(seg_valid_vec1) if len(seg_valid_vec1) > 0 else 0
        }
        
        segment_analysis.append(segment_info)
        
        # 判断是否为完全相同的段（相似度>=0.99且相同元素比例>=0.95）
        if seg_sim is not None and seg_sim >= 0.99 and segment_info['identical_ratio'] >= 0.95:
            identical_segments.append(segment_info)
        # 判断是否为差异明显的段（相似度<0.8或相同元素比例<0.5）
        elif seg_sim is not None and (seg_sim < 0.8 or segment_info['identical_ratio'] < 0.5):
            different_segments.append(segment_info)
    
    # 计算整体统计
    total_identical = sum(seg['identical_count'] for seg in segment_analysis if seg['similarity'] is not None)
    total_different = sum(seg['different_count'] for seg in segment_analysis if seg['similarity'] is not None)
    total_valid = total_identical + total_different
    
    return {
        'total_length': min_len,
        'valid_length': total_valid,
        'identical_count': total_identical,
        'different_count': total_different,
        'identical_ratio': total_identical / total_valid if total_valid > 0 else 0,
        'identical_segments': identical_segments,
        'different_segments': different_segments,
        'segment_analysis': segment_analysis,
        'num_segments': num_segments
    }


def print_similarity_analysis(analysis_result, vec1_name="vec1", vec2_name="vec2", max_segments_to_show=10):
    """
    打印相似度分析结果
    """
    print(f"\n{'='*100}")
    print(f"向量相似度分析: {vec1_name} vs {vec2_name}")
    print(f"{'='*100}")
    print(f"总长度: {analysis_result['total_length']}")
    print(f"有效长度: {analysis_result['valid_length']}")
    print(f"完全相同元素数: {analysis_result['identical_count']}")
    print(f"不同元素数: {analysis_result['different_count']}")
    print(f"相同元素比例: {analysis_result['identical_ratio']:.4f} ({analysis_result['identical_ratio']*100:.2f}%)")
    print(f"\n总段数: {analysis_result['num_segments']}")
    print(f"完全相同的段数: {len(analysis_result['identical_segments'])}")
    print(f"差异明显的段数: {len(analysis_result['different_segments'])}")
    
    # 显示完全相同的段
    if analysis_result['identical_segments']:
        print(f"\n【完全相同的段】（前{min(max_segments_to_show, len(analysis_result['identical_segments']))}个）:")
        print("-" * 100)
        print(f"{'段号':<8} {'起始位置':<12} {'结束位置':<12} {'长度':<8} {'有效长度':<10} {'相似度':<12} {'相同比例':<12}")
        print("-" * 100)
        for seg in analysis_result['identical_segments'][:max_segments_to_show]:
            print(f"{seg['segment']:<8} {seg['start']:<12} {seg['end']:<12} {seg['length']:<8} "
                  f"{seg['valid_length']:<10} {seg['similarity']:<12.6f} {seg['identical_ratio']:<12.4f}")
    
    # 显示差异明显的段
    if analysis_result['different_segments']:
        print(f"\n【差异明显的段】（前{min(max_segments_to_show, len(analysis_result['different_segments']))}个）:")
        print("-" * 100)
        print(f"{'段号':<8} {'起始位置':<12} {'结束位置':<12} {'长度':<8} {'有效长度':<10} {'相似度':<12} {'相同比例':<12}")
        print("-" * 100)
        for seg in analysis_result['different_segments'][:max_segments_to_show]:
            print(f"{seg['segment']:<8} {seg['start']:<12} {seg['end']:<12} {seg['length']:<8} "
                  f"{seg['valid_length']:<10} {seg['similarity']:<12.6f} {seg['identical_ratio']:<12.4f}")
    
    # 显示所有段的统计信息
    print(f"\n【所有段的相似度统计】:")
    print("-" * 100)
    similarities = [seg['similarity'] for seg in analysis_result['segment_analysis'] if seg['similarity'] is not None]
    if similarities:
        print(f"平均相似度: {np.mean(similarities):.6f}")
        print(f"最大相似度: {np.max(similarities):.6f}")
        print(f"最小相似度: {np.min(similarities):.6f}")
        print(f"中位数相似度: {np.median(similarities):.6f}")
        print(f"相似度>=0.9的段数: {sum(1 for s in similarities if s >= 0.9)}")
        print(f"相似度<0.5的段数: {sum(1 for s in similarities if s < 0.5)}")
    print(f"{'='*100}\n")


def cosine_similarity(vec1, vec2, analyze=False, vec1_name="vec1", vec2_name="vec2"):
    """
    计算两个向量的余弦相似度
    在计算前对向量进行预处理：
    1. 首先确保两个向量长度一致：在短的向量后面补0
    2. 第一个向量在前面补0，第二个向量在前面补1
    3. 补的长度为原向量长度的1%
    4. 如果补值后两个向量长度不一致，继续补齐较短的向量
    过滤掉值为-1的位置（只有当vec1和vec2都为-1时，才过滤掉）
    """
    # 保存原始向量的副本（用于分析）
    orig_vec1 = vec1.copy()
    orig_vec2 = vec2.copy()
    
    # 首先确保两个向量长度一致：在短的向量后面补0
    if len(vec1) < len(vec2):
        # vec1较短，在后面补0
        padding = np.zeros(len(vec2) - len(vec1), dtype=vec1.dtype)
        vec1 = np.concatenate([vec1, padding])
    elif len(vec2) < len(vec1):
        # vec2较短，在后面补0
        padding = np.zeros(len(vec1) - len(vec2), dtype=vec2.dtype)
        vec2 = np.concatenate([vec2, padding])
    
    # 保存原始向量长度（补0后的长度）
    orig_len1 = len(vec1)
    orig_len2 = len(vec2)
    
    # # 计算需要补的长度：原向量长度的1%（向上取整）
    # pad_len1 = int(np.ceil(orig_len1 * 0.01))
    # pad_len2 = int(np.ceil(orig_len2 * 0.01))
    
    # # 对vec1在前面补0
    # if pad_len1 > 0:
    #     pad1 = np.zeros(pad_len1, dtype=vec1.dtype)
    #     vec1 = np.concatenate([pad1, vec1])
    
    # # 对vec2在前面补1
    # if pad_len2 > 0:
    #     pad2 = np.ones(pad_len2, dtype=vec2.dtype)
    #     vec2 = np.concatenate([pad2, vec2])
    
    # # 如果两个向量长度不一致，继续补齐较短的向量
    # if len(vec1) < len(vec2):
    #     # vec1较短，继续在前面补0
    #     additional_pad = np.zeros(len(vec2) - len(vec1), dtype=vec1.dtype)
    #     vec1 = np.concatenate([additional_pad, vec1])
    # elif len(vec2) < len(vec1):
    #     # vec2较短，继续在前面补1
    #     additional_pad = np.ones(len(vec1) - len(vec2), dtype=vec2.dtype)
    #     vec2 = np.concatenate([additional_pad, vec2])
    
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
    
    # 如果需要分析，进行详细分析
    if analyze:
        analysis_result = analyze_vector_similarity(orig_vec1, orig_vec2, 
                                                     vec1_name=vec1_name, vec2_name=vec2_name)
        print_similarity_analysis(analysis_result, vec1_name=vec1_name, vec2_name=vec2_name)
    
    return similarity, None


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


def count_unique_vectors(vectors):
    """
    统计向量集合中唯一向量的种类数量
    一模一样的向量算作一类
    
    参数:
        vectors: 向量列表，每个向量是numpy数组
    
    返回:
        unique_count: 唯一向量的种类数量
    """
    if not vectors:
        return 0
    
    # 将向量转换为可哈希的元组形式
    vector_tuples = []
    for vec in vectors:
        try:
            # 确保是numpy数组
            if not isinstance(vec, np.ndarray):
                vec = np.array(vec)
            
            # 转换为元组以便比较（使用tolist()确保可以正确比较）
            vec_tuple = tuple(vec.tolist())
            vector_tuples.append(vec_tuple)
        except Exception as e:
            print(f"警告: 无法处理向量类型 {type(vec)}, 错误: {e}")
            continue
    
    # 使用set去重，统计唯一向量的数量
    unique_vectors = set(vector_tuples)
    return len(unique_vectors)


def load_vectors(iteration_range):
    """
    加载所有迭代的覆盖向量
    返回原始种子和变异种子的向量字典
    """
    ori_cov_dir = "/media/lzq/D/lzq/pylot_test/pylot/ori_cov_vector"
    mut_cov_dir = "/media/lzq/D/lzq/pylot_test/pylot/cov_vector"
    ori_seeds_dir = "/media/lzq/D/lzq/pylot_test/pylot/ori_seeds_vectors"
    mut_seeds_dir = "/media/lzq/D/lzq/pylot_test/pylot/error_seeds_vectors"
    
    module_names = ['perception_vector', 'planning_vector', 'control_vector', 'other_vector', 'nac_vector']
    
    ori_vectors = {}  # {iteration: {module_name: vector}}
    mut_vectors = {}  # {iteration: {module_name: vector}}
    
    print("=" * 100)
    print("加载覆盖向量")
    print("=" * 100)
    
    for iteration in iteration_range:
        ori_vectors[iteration] = {}
        mut_vectors[iteration] = {}
        
        # 读取原始种子代码覆盖向量
        ori_cov_file = os.path.join(ori_cov_dir, f"{iteration}_array_vector.npy")
        if os.path.exists(ori_cov_file):
            data = read_npy_file(ori_cov_file)
            if data and isinstance(data, dict):
                for module_name in ['total_vector', 'perception_vector', 'planning_vector', 'control_vector', 'other_vector']:
                    if module_name in data:
                        ori_vectors[iteration][module_name] = data[module_name]
        else:
            print(f"警告: 原始种子迭代 {iteration} 的代码覆盖向量文件不存在")
        
        # 读取变异种子代码覆盖向量
        mut_cov_file = os.path.join(mut_cov_dir, f"{iteration}_array_vector.npy")
        if os.path.exists(mut_cov_file):
            data = read_npy_file(mut_cov_file)
            if data and isinstance(data, dict):
                for module_name in ['total_vector', 'perception_vector', 'planning_vector', 'control_vector', 'other_vector']:
                    if module_name in data:
                        mut_vectors[iteration][module_name] = data[module_name]
        else:
            print(f"警告: 变异种子迭代 {iteration} 的代码覆盖向量文件不存在")
        
        # 读取原始种子神经元覆盖向量（注意是i-1）
        ori_seeds_file = os.path.join(ori_seeds_dir, f"{iteration-1}.pickle")
        if os.path.exists(ori_seeds_file):
            data = read_pickle_file(ori_seeds_file)
            if data and isinstance(data, dict) and 'nac_vector' in data:
                nac_vector = data['nac_vector']
                original_len = len(nac_vector) if isinstance(nac_vector, (list, np.ndarray)) else 0
                if isinstance(nac_vector, list):
                    nac_vector = np.array(nac_vector)
                # 只取300-400这一段（索引300到399，共100个元素）
                if len(nac_vector) > 400:
                    nac_vector = nac_vector[300:400]
                elif len(nac_vector) > 300:
                    # 如果向量长度不足400，只取到末尾
                    nac_vector = nac_vector[300:]
                else:
                    # 如果向量长度不足300，使用空向量
                    nac_vector = np.array([])
                    print(f"警告: 原始种子迭代 {iteration} 的nac_vector长度不足300（实际长度: {original_len}）")
                ori_vectors[iteration]['nac_vector'] = nac_vector
        else:
            print(f"警告: 原始种子迭代 {iteration} 的神经元覆盖向量文件不存在: {ori_seeds_file}")
        
        # 读取变异种子神经元覆盖向量
        mut_error_file = os.path.join(mut_seeds_dir, f"{iteration}_error.pickle")
        mut_normal_file = os.path.join(mut_seeds_dir, f"{iteration}_normal.pickle")
        
        nac_vector = None
        if os.path.exists(mut_error_file):
            data = read_pickle_file(mut_error_file)
            if data and isinstance(data, dict) and 'nac_vector' in data:
                nac_vector = data['nac_vector']
        elif os.path.exists(mut_normal_file):
            data = read_pickle_file(mut_normal_file)
            if data and isinstance(data, dict) and 'nac_vector' in data:
                nac_vector = data['nac_vector']
        
        if nac_vector is not None:
            if isinstance(nac_vector, list):
                nac_vector = np.array(nac_vector)
            # 只取300-400这一段（索引300到399，共100个元素）
            original_len = len(nac_vector)
            if len(nac_vector) > 400:
                nac_vector = nac_vector[300:400]
            elif len(nac_vector) > 300:
                # 如果向量长度不足400，只取到末尾
                nac_vector = nac_vector[300:]
            else:
                # 如果向量长度不足300，使用空向量
                nac_vector = np.array([])
                print(f"警告: 变异种子迭代 {iteration} 的nac_vector长度不足300（实际长度: {original_len}）")
            mut_vectors[iteration]['nac_vector'] = nac_vector
        else:
            print(f"警告: 变异种子迭代 {iteration} 的神经元覆盖向量文件不存在")
    
    print(f"成功加载原始种子 {len(ori_vectors)} 个迭代的覆盖向量")
    print(f"成功加载变异种子 {len(mut_vectors)} 个迭代的覆盖向量")
    
    return ori_vectors, mut_vectors


def main():
    iteration_range = range(1, 101)  # 1-100
    
    # 模块向量名称
    module_names = ['perception_vector', 'planning_vector', 'control_vector', 'other_vector', 'nac_vector']
    
    # 1. 加载所有向量
    ori_vectors, mut_vectors = load_vectors(iteration_range)
    
    # 2. 计算原始种子内部各模块的余弦相似度
    print("\n" + "=" * 100)
    print("步骤1: 计算原始种子内部各模块覆盖向量的余弦相似度")
    print("=" * 100)
    
    ori_within_results = {}
    
    for module_name in module_names:
        print(f"\n{module_name}:")
        print("-" * 100)
        
        # 收集所有原始种子的该模块向量
        vectors = []
        for iteration in iteration_range:
            if iteration in ori_vectors and module_name in ori_vectors[iteration]:
                vectors.append(ori_vectors[iteration][module_name])
        
        if len(vectors) < 2:
            print(f"  向量数量少于2个（{len(vectors)}），无法计算")
            ori_within_results[module_name] = None
            continue
        
        # 计算组内相似度
        avg_sim, max_sim, min_sim, median_sim, sim_list, error_msg = calculate_within_group_similarity(vectors)
        
        if error_msg:
            print(f"  计算失败 - {error_msg}")
            ori_within_results[module_name] = None
        else:
            print(f"  向量数量: {len(vectors)}")
            print(f"  两两组合数: {len(sim_list)}")
            print(f"  平均相似度: {avg_sim:.6f}")
            print(f"  最大相似度: {max_sim:.6f}")
            print(f"  最小相似度: {min_sim:.6f}")
            print(f"  中位数相似度: {median_sim:.6f}")
            ori_within_results[module_name] = {
                'avg': avg_sim,
                'max': max_sim,
                'min': min_sim,
                'median': median_sim,
                'count': len(sim_list)
            }
    
    # 3. 计算变异种子内部各模块的余弦相似度
    print("\n" + "=" * 100)
    print("步骤2: 计算变异种子内部各模块覆盖向量的余弦相似度")
    print("=" * 100)
    
    mut_within_results = {}
    
    for module_name in module_names:
        print(f"\n{module_name}:")
        print("-" * 100)
        
        # 收集所有变异种子的该模块向量
        vectors = []
        for iteration in iteration_range:
            if iteration in mut_vectors and module_name in mut_vectors[iteration]:
                vectors.append(mut_vectors[iteration][module_name])
        
        if len(vectors) < 2:
            print(f"  向量数量少于2个（{len(vectors)}），无法计算")
            mut_within_results[module_name] = None
            continue
        
        # 计算组内相似度
        avg_sim, max_sim, min_sim, median_sim, sim_list, error_msg = calculate_within_group_similarity(vectors)
        
        if error_msg:
            print(f"  计算失败 - {error_msg}")
            mut_within_results[module_name] = None
        else:
            print(f"  向量数量: {len(vectors)}")
            print(f"  两两组合数: {len(sim_list)}")
            print(f"  平均相似度: {avg_sim:.6f}")
            print(f"  最大相似度: {max_sim:.6f}")
            print(f"  最小相似度: {min_sim:.6f}")
            print(f"  中位数相似度: {median_sim:.6f}")
            mut_within_results[module_name] = {
                'avg': avg_sim,
                'max': max_sim,
                'min': min_sim,
                'median': median_sim,
                'count': len(sim_list)
            }
    
    # 4. 计算原始种子和变异种子之间的余弦相似度
    print("\n" + "=" * 100)
    print("步骤3: 计算原始种子和变异种子覆盖向量之间的余弦相似度")
    print("=" * 100)
    
    ori_mut_results = {}
    
    for module_name in module_names:
        print(f"\n{module_name}:")
        print("-" * 100)
        
        # 收集原始种子的该模块向量
        ori_module_vectors = []
        for iteration in iteration_range:
            if iteration in ori_vectors and module_name in ori_vectors[iteration]:
                ori_module_vectors.append(ori_vectors[iteration][module_name])
        
        # 收集变异种子的该模块向量
        mut_module_vectors = []
        for iteration in iteration_range:
            if iteration in mut_vectors and module_name in mut_vectors[iteration]:
                mut_module_vectors.append(mut_vectors[iteration][module_name])
        
        if not ori_module_vectors or not mut_module_vectors:
            print(f"  无法计算（原始种子: {len(ori_module_vectors)} 个向量，变异种子: {len(mut_module_vectors)} 个向量）")
            ori_mut_results[module_name] = None
            continue
        
        # 计算两组之间的相似度
        avg_sim, max_sim, min_sim, median_sim, sim_list, error_msg = calculate_between_groups_similarity(
            ori_module_vectors, mut_module_vectors
        )
        
        if error_msg:
            print(f"  计算失败 - {error_msg}")
            ori_mut_results[module_name] = None
        else:
            print(f"  原始种子向量数: {len(ori_module_vectors)}")
            print(f"  变异种子向量数: {len(mut_module_vectors)}")
            print(f"  两两组合数: {len(sim_list)}")
            print(f"  平均相似度: {avg_sim:.6f}")
            print(f"  最大相似度: {max_sim:.6f}")
            print(f"  最小相似度: {min_sim:.6f}")
            print(f"  中位数相似度: {median_sim:.6f}")
            ori_mut_results[module_name] = {
                'avg': avg_sim,
                'max': max_sim,
                'min': min_sim,
                'median': median_sim,
                'count': len(sim_list)
            }
    
    # 5. 打印汇总结果
    print("\n" + "=" * 100)
    print("步骤4: 汇总结果")
    print("=" * 100)
    
    print("\n【原始种子内部各模块余弦相似度 - 平均值】")
    print("-" * 100)
    print(f"{'模块':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    row = f"{'原始种子':<25}"
    for module_name in module_names:
        result = ori_within_results.get(module_name)
        if result is not None and isinstance(result, dict) and 'avg' in result:
            row += f"{result['avg']:<15.6f}"
        else:
            row += f"{'N/A':<15}"
    print(row)
    
    print("\n【原始种子内部各模块余弦相似度 - 最大值】")
    print("-" * 100)
    print(f"{'模块':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    row = f"{'原始种子':<25}"
    for module_name in module_names:
        result = ori_within_results.get(module_name)
        if result is not None and isinstance(result, dict) and 'max' in result:
            row += f"{result['max']:<15.6f}"
        else:
            row += f"{'N/A':<15}"
    print(row)
    
    print("\n【原始种子内部各模块余弦相似度 - 最小值】")
    print("-" * 100)
    print(f"{'模块':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    row = f"{'原始种子':<25}"
    for module_name in module_names:
        result = ori_within_results.get(module_name)
        if result is not None and isinstance(result, dict) and 'min' in result:
            row += f"{result['min']:<15.6f}"
        else:
            row += f"{'N/A':<15}"
    print(row)
    
    print("\n【原始种子内部各模块余弦相似度 - 中位数】")
    print("-" * 100)
    print(f"{'模块':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    row = f"{'原始种子':<25}"
    for module_name in module_names:
        result = ori_within_results.get(module_name)
        if result is not None and isinstance(result, dict) and 'median' in result:
            row += f"{result['median']:<15.6f}"
        else:
            row += f"{'N/A':<15}"
    print(row)
    
    print("\n【变异种子内部各模块余弦相似度 - 平均值】")
    print("-" * 100)
    print(f"{'模块':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    row = f"{'变异种子':<25}"
    for module_name in module_names:
        result = mut_within_results.get(module_name)
        if result is not None and isinstance(result, dict) and 'avg' in result:
            row += f"{result['avg']:<15.6f}"
        else:
            row += f"{'N/A':<15}"
    print(row)
    
    print("\n【变异种子内部各模块余弦相似度 - 最大值】")
    print("-" * 100)
    print(f"{'模块':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    row = f"{'变异种子':<25}"
    for module_name in module_names:
        result = mut_within_results.get(module_name)
        if result is not None and isinstance(result, dict) and 'max' in result:
            row += f"{result['max']:<15.6f}"
        else:
            row += f"{'N/A':<15}"
    print(row)
    
    print("\n【变异种子内部各模块余弦相似度 - 最小值】")
    print("-" * 100)
    print(f"{'模块':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    row = f"{'变异种子':<25}"
    for module_name in module_names:
        result = mut_within_results.get(module_name)
        if result is not None and isinstance(result, dict) and 'min' in result:
            row += f"{result['min']:<15.6f}"
        else:
            row += f"{'N/A':<15}"
    print(row)
    
    print("\n【变异种子内部各模块余弦相似度 - 中位数】")
    print("-" * 100)
    print(f"{'模块':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    row = f"{'变异种子':<25}"
    for module_name in module_names:
        result = mut_within_results.get(module_name)
        if result is not None and isinstance(result, dict) and 'median' in result:
            row += f"{result['median']:<15.6f}"
        else:
            row += f"{'N/A':<15}"
    print(row)
    
    print("\n【原始种子和变异种子之间的余弦相似度 - 平均值】")
    print("-" * 100)
    print(f"{'模块':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    row = f"{'原始vs变异':<25}"
    for module_name in module_names:
        result = ori_mut_results.get(module_name)
        if result is not None and isinstance(result, dict) and 'avg' in result:
            row += f"{result['avg']:<15.6f}"
        else:
            row += f"{'N/A':<15}"
    print(row)
    
    print("\n【原始种子和变异种子之间的余弦相似度 - 最大值】")
    print("-" * 100)
    print(f"{'模块':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    row = f"{'原始vs变异':<25}"
    for module_name in module_names:
        result = ori_mut_results.get(module_name)
        if result is not None and isinstance(result, dict) and 'max' in result:
            row += f"{result['max']:<15.6f}"
        else:
            row += f"{'N/A':<15}"
    print(row)
    
    print("\n【原始种子和变异种子之间的余弦相似度 - 最小值】")
    print("-" * 100)
    print(f"{'模块':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    row = f"{'原始vs变异':<25}"
    for module_name in module_names:
        result = ori_mut_results.get(module_name)
        if result is not None and isinstance(result, dict) and 'min' in result:
            row += f"{result['min']:<15.6f}"
        else:
            row += f"{'N/A':<15}"
    print(row)
    
    print("\n【原始种子和变异种子之间的余弦相似度 - 中位数】")
    print("-" * 100)
    print(f"{'模块':<25} {'perception':<15} {'planning':<15} {'control':<15} {'other':<15} {'nac':<15}")
    print("-" * 100)
    row = f"{'原始vs变异':<25}"
    for module_name in module_names:
        result = ori_mut_results.get(module_name)
        if result is not None and isinstance(result, dict) and 'median' in result:
            row += f"{result['median']:<15.6f}"
        else:
            row += f"{'N/A':<15}"
    print(row)
    
    print("=" * 100)
    
    # 6. 统计原始种子和变异种子中total_vector和nac_vector的种类数量
    print("\n" + "=" * 100)
    print("步骤5: 统计原始种子和变异种子中total_vector和nac_vector的种类数量")
    print("=" * 100)
    
    # 收集原始种子的total_vector和nac_vector
    ori_total_vectors = []
    ori_nac_vectors = []
    for iteration in iteration_range:
        if iteration in ori_vectors:
            if 'total_vector' in ori_vectors[iteration]:
                ori_total_vectors.append(ori_vectors[iteration]['total_vector'])
            if 'nac_vector' in ori_vectors[iteration]:
                ori_nac_vectors.append(ori_vectors[iteration]['nac_vector'])
    
    # 收集变异种子的total_vector和nac_vector
    mut_total_vectors = []
    mut_nac_vectors = []
    for iteration in iteration_range:
        if iteration in mut_vectors:
            if 'total_vector' in mut_vectors[iteration]:
                mut_total_vectors.append(mut_vectors[iteration]['total_vector'])
            if 'nac_vector' in mut_vectors[iteration]:
                mut_nac_vectors.append(mut_vectors[iteration]['nac_vector'])
    
    # 统计种类数量
    ori_total_pattern_count = count_unique_vectors(ori_total_vectors)
    ori_nac_pattern_count = count_unique_vectors(ori_nac_vectors)
    mut_total_pattern_count = count_unique_vectors(mut_total_vectors)
    mut_nac_pattern_count = count_unique_vectors(mut_nac_vectors)
    
    print(f"\n原始种子统计:")
    print(f"  total_vector向量总数: {len(ori_total_vectors)}")
    print(f"  total_vector种类数量: {ori_total_pattern_count}")
    print(f"  nac_vector向量总数: {len(ori_nac_vectors)}")
    print(f"  nac_vector种类数量: {ori_nac_pattern_count}")
    
    print(f"\n变异种子统计:")
    print(f"  total_vector向量总数: {len(mut_total_vectors)}")
    print(f"  total_vector种类数量: {mut_total_pattern_count}")
    print(f"  nac_vector向量总数: {len(mut_nac_vectors)}")
    print(f"  nac_vector种类数量: {mut_nac_pattern_count}")
    
    # 输出结果表
    print("\n" + "=" * 100)
    print("结果表: 向量种类数量统计")
    print("=" * 100)
    print(f"{'':<20} {'total_code_pattern':<25} {'nac_pattern':<25}")
    print("-" * 100)
    print(f"{'原始种子':<20} {ori_total_pattern_count:<25} {ori_nac_pattern_count:<25}")
    print(f"{'变异生成':<20} {mut_total_pattern_count:<25} {mut_nac_pattern_count:<25}")
    print("=" * 100)


if __name__ == "__main__":
    main()
