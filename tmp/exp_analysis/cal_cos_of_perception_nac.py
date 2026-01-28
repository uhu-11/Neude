'''
判断是否发生cur_model_errors：
读取/home/lzq/test_reslut/会议论文实验结果/exp35/100.json文件，取出ious列的数据，包含一百次迭代的ious，每次迭代的ious都是一个数组，包含十个元素，按位置对应了十个测试用例的测试值。
正确的判断标准如下：
ious：感知模块IoU > 0.5 为正确，否则为错误。
根据上面的判断标准判断感知模块是否发生错误，如果发生错误，则认为该测试用例发生cur_model_errors。

按照每个测试用例是否发生cur_model_errors，将100次迭代所包含的1000个测试用例分为两组：一组是发生cur_model_errors的测试用例，另一组是未发生cur_model_errors的测试用例。
再计算这两组测试用例之间的覆盖向量nac_vector的余弦相似度。

1.计算"没有发生cur_model_errors的测试用例"之间的覆盖向量nac_vector的余弦相似度。

2.计算"发生cur_model_errors的测试用例"和"没有发生cur_model_errors的测试用例"之间的覆盖向量nac_vector的余弦相似度。

3.计算所有"发生cur_model_errors的测试用例"之间的覆盖向量nac_vector的余弦相似度。

'''

import json
import numpy as np
import prettytable

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    """
    # 检查向量长度是否一致
    if len(vec1) != len(vec2):
        return None, f"向量长度不一致: {len(vec1)} vs {len(vec2)}"
    
    # 转换为numpy数组
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    
    # 计算向量的模
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # 避免除零
    if norm1 == 0 or norm2 == 0:
        return 0.0, "其中一个向量为零向量"
    
    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
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


def main():
    # 文件路径
    # json_file_path = '/home/lzq/test_reslut/会议论文实验结果/exp35/100.json'
    json_file_path = '/home/lzq/result/datas/100_nac_vectors_exp35.json'
    
    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败 {json_file_path}: {e}")
        return
    
    # 获取batch_nac_vectors和ious
    batch_nac_vectors = data.get('batch_nac_vectors', [])
    ious = data.get('ious', [])
    
    print(f"读取到 {len(batch_nac_vectors)} 次迭代的神经元向量")
    print(f"读取到 {len(ious)} 次迭代的IoU值")
    
    # 判断标准
    IOU_THRESHOLD = 0.5  # ious > 0.5 为正确
    
    # 按照测试用例级别分组
    # 收集所有测试用例的nac_vector和错误状态
    test_cases_with_error = []  # 发生错误的测试用例的nac_vector列表
    test_cases_without_error = []  # 没有错误的测试用例的nac_vector列表
    
    total_test_cases = 0
    error_test_cases = 0
    no_error_test_cases = 0
    
    # 遍历所有迭代
    for iter_idx in range(len(batch_nac_vectors)):
        iter_nac_vectors = batch_nac_vectors[iter_idx] if iter_idx < len(batch_nac_vectors) else []
        iter_ious = ious[iter_idx] if iter_idx < len(ious) else []
        
        # 遍历该迭代的10个测试用例
        num_cases = len(iter_nac_vectors) if iter_nac_vectors else 0
        
        for case_idx in range(num_cases):
            # 获取该测试用例的nac_vector
            nac_vector = iter_nac_vectors[case_idx] if case_idx < len(iter_nac_vectors) else None
            
            # 取出每个向量的前2145长度
            if nac_vector is not None:
                nac_vector = np.array(nac_vector)
                if len(nac_vector) > 900:
                    nac_vector = nac_vector[800:900]
                elif len(nac_vector) < 900:
                    # 如果向量长度小于2145，用0填充
                    padding = np.zeros(900 - len(nac_vector), dtype=nac_vector.dtype)
                    nac_vector = np.concatenate([nac_vector, padding])
                    nac_vector = nac_vector[800:900]
            
            # 获取该测试用例的IoU值
            iou_val = iter_ious[case_idx] if case_idx < len(iter_ious) else None
            
            # 判断是否发生错误
            has_error = False
            if iou_val is not None:
                if iou_val <= IOU_THRESHOLD:
                    has_error = True
            
            # 如果nac_vector有效，则添加到对应的组
            if nac_vector is not None:
                total_test_cases += 1
                if has_error:
                    test_cases_with_error.append(nac_vector)
                    error_test_cases += 1
                else:
                    test_cases_without_error.append(nac_vector)
                    no_error_test_cases += 1
    
    print(f"\n总共处理了 {total_test_cases} 个测试用例")
    print(f"发生cur_model_errors的测试用例数量: {error_test_cases}")
    print(f"没有发生cur_model_errors的测试用例数量: {no_error_test_cases}")
    
    # 1. 计算"没有发生cur_model_errors的测试用例"之间的余弦相似度
    print("\n" + "="*80)
    print("1. 计算\"没有发生cur_model_errors的测试用例\"之间的覆盖向量nac_vector的余弦相似度")
    print("="*80)
    
    if len(test_cases_without_error) >= 2:
        avg_sim, max_sim, min_sim, median_sim, sim_list, error_msg = calculate_within_group_similarity(
            test_cases_without_error
        )
        
        if error_msg:
            print(f"计算失败: {error_msg}")
        else:
            print(f"平均相似度: {avg_sim:.4f}")
            print(f"最大相似度: {max_sim:.4f}")
            print(f"最小相似度: {min_sim:.4f}")
            print(f"中位数相似度: {median_sim:.4f}")
            print(f"相似度对数量: {len(sim_list)}")
    else:
        print(f"没有错误的测试用例数量不足（需要至少2个），当前数量: {len(test_cases_without_error)}")
    
    # 2. 计算"发生cur_model_errors的测试用例"和"没有发生cur_model_errors的测试用例"之间的余弦相似度
    print("\n" + "="*80)
    print("2. 计算\"发生cur_model_errors的测试用例\"和\"没有发生cur_model_errors的测试用例\"之间的覆盖向量nac_vector的余弦相似度")
    print("="*80)
    
    if test_cases_with_error and test_cases_without_error:
        avg_sim, max_sim, min_sim, median_sim, sim_list, error_msg = calculate_between_groups_similarity(
            test_cases_with_error, test_cases_without_error
        )
        
        if error_msg:
            print(f"计算失败: {error_msg}")
        else:
            print(f"平均相似度: {avg_sim:.4f}")
            print(f"最大相似度: {max_sim:.4f}")
            print(f"最小相似度: {min_sim:.4f}")
            print(f"中位数相似度: {median_sim:.4f}")
            print(f"相似度对数量: {len(sim_list)}")
    else:
        print(f"错误测试用例数量: {len(test_cases_with_error)}, 无错误测试用例数量: {len(test_cases_without_error)}")
        if not test_cases_with_error:
            print("没有发生错误的测试用例，无法计算")
        if not test_cases_without_error:
            print("没有无错误的测试用例，无法计算")
    
    # 3. 计算所有"发生cur_model_errors的测试用例"之间的余弦相似度
    print("\n" + "="*80)
    print("3. 计算所有\"发生cur_model_errors的测试用例\"之间的覆盖向量nac_vector的余弦相似度")
    print("="*80)
    
    if len(test_cases_with_error) >= 2:
        avg_sim, max_sim, min_sim, median_sim, sim_list, error_msg = calculate_within_group_similarity(
            test_cases_with_error
        )
        
        if error_msg:
            print(f"计算失败: {error_msg}")
        else:
            print(f"平均相似度: {avg_sim:.4f}")
            print(f"最大相似度: {max_sim:.4f}")
            print(f"最小相似度: {min_sim:.4f}")
            print(f"中位数相似度: {median_sim:.4f}")
            print(f"相似度对数量: {len(sim_list)}")
    else:
        print(f"发生错误的测试用例数量不足（需要至少2个），当前数量: {len(test_cases_with_error)}")
    
    # 汇总表格
    print("\n" + "="*80)
    print("汇总统计表")
    print("="*80)
    
    table = prettytable.PrettyTable()
    table.field_names = ["相似度类型", "平均值", "最大值", "最小值", "中位数", "相似度对数量"]
    
    # 1. 没有错误的测试用例之间的相似度
    if len(test_cases_without_error) >= 2:
        avg_sim, max_sim, min_sim, median_sim, sim_list, error_msg = calculate_within_group_similarity(
            test_cases_without_error
        )
        if error_msg is None:
            table.add_row([
                "没有错误的测试用例之间",
                f"{avg_sim:.4f}",
                f"{max_sim:.4f}",
                f"{min_sim:.4f}",
                f"{median_sim:.4f}",
                len(sim_list)
            ])
    
    # 2. 有错误和无错误测试用例之间的相似度
    if test_cases_with_error and test_cases_without_error:
        avg_sim, max_sim, min_sim, median_sim, sim_list, error_msg = calculate_between_groups_similarity(
            test_cases_with_error, test_cases_without_error
        )
        if error_msg is None:
            table.add_row([
                "有错误和无错误测试用例之间",
                f"{avg_sim:.4f}",
                f"{max_sim:.4f}",
                f"{min_sim:.4f}",
                f"{median_sim:.4f}",
                len(sim_list)
            ])
    
    # 3. 有错误的测试用例之间的相似度
    if len(test_cases_with_error) >= 2:
        avg_sim, max_sim, min_sim, median_sim, sim_list, error_msg = calculate_within_group_similarity(
            test_cases_with_error
        )
        if error_msg is None:
            table.add_row([
                "有错误的测试用例之间",
                f"{avg_sim:.4f}",
                f"{max_sim:.4f}",
                f"{min_sim:.4f}",
                f"{median_sim:.4f}",
                len(sim_list)
            ])
    
    print(table)


if __name__ == "__main__":
    main()
