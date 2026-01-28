
import os
import pickle
import math
from collections import defaultdict
from itertools import combinations
from datetime import datetime

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    # 检查向量长度是否一致
    if len(vec1) != len(vec2):
        return None, f"向量长度不一致: {len(vec1)} vs {len(vec2)}"
    
    # 计算点积
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # 计算向量的模
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    # 避免除零
    if norm1 == 0 or norm2 == 0:
        return 0.0, "其中一个向量为零向量"
    
    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
    return similarity, None

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

def read_vector_file(file_path):
    """读取向量文件"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"读取向量文件失败 {file_path}: {e}")
        return None

def calculate_similarities_for_group(vectors_data, vector_type='line_vector'):
    """计算一组向量之间的相似度"""
    similarities = []
    length_info = {}
    invalid_pairs = []
    
    # 检查向量长度
    for iter_num, data in vectors_data.items():
        length_info[iter_num] = len(data[vector_type])
    
    # 计算所有向量对之间的相似度
    for (iter1, iter2) in combinations(sorted(vectors_data.keys(), key=int), 2):
        vec1 = vectors_data[iter1][vector_type]
        vec2 = vectors_data[iter2][vector_type]
        similarity, error_msg = cosine_similarity(vec1, vec2)
        
        if error_msg:
            invalid_pairs.append((iter1, iter2, error_msg))
        else:
            similarities.append(similarity)
    
    return similarities, length_info, invalid_pairs

def main():
    error_infos_dir = 'pylot/error_infos'
    vectors_dir = 'error_seeds_vectors'
    output_file = 'similarity_analysis_result.txt'
    
    # 按错误签名分组
    error_groups = defaultdict(list)
    
    # 首先从向量目录中读取所有 *_error.pickle 文件，提取迭代序号
    error_iterations = set()
    if os.path.exists(vectors_dir):
        for filename in os.listdir(vectors_dir):
            if filename.endswith('_error.pickle'):
                # 从文件名中提取迭代序号，例如从 "2_error.pickle" 提取 "2"
                iteration_num = filename.replace('_error.pickle', '')
                error_iterations.add(iteration_num)
    
    print(f"从向量目录中找到 {len(error_iterations)} 个错误用例\n")
    
    # 读取对应的错误信息文件并分组
    if os.path.exists(error_infos_dir):
        for iteration_num in error_iterations:
            error_file = os.path.join(error_infos_dir, f"{iteration_num}.txt")
            if os.path.exists(error_file):
                error_content = read_error_info(error_file)
                if error_content:
                    error_type, error_signature = extract_error_signature(error_content)
                    if error_signature:
                        error_groups[error_signature].append(iteration_num)
            else:
                # 如果没有对应的错误信息文件，仍然可以处理（可能错误信息在其他地方）
                # 使用一个默认的错误签名
                error_signature = "未知错误"
                error_groups[error_signature].append(iteration_num)
    
    print(f"找到 {len(error_groups)} 种不同的错误类型\n")
    
    # 收集所有错误组的向量数据
    all_error_vectors = {}  # {error_signature: {iter_num: {line_vector, nac_vector}}}
    
    # 先读取所有向量数据
    for error_signature, iteration_nums in error_groups.items():
        vectors_data = {}
        for iter_num in iteration_nums:
            # 新的文件命名格式：{iter_num}_error.pickle
            vector_file = os.path.join(vectors_dir, f"{iter_num}_error.pickle")
            if os.path.exists(vector_file):
                data = read_vector_file(vector_file)
                if data and 'line_vector' in data and 'nac_vector' in data:
                    vectors_data[iter_num] = {
                        'line_vector': data['line_vector'],
                        'nac_vector': data['nac_vector']
                    }
        if vectors_data:
            all_error_vectors[error_signature] = vectors_data
    
    # 准备输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("覆盖向量相似度分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")
        
        # 统计相同错误内部的相似度
        same_error_line_similarities = []
        same_error_nac_similarities = []
        
        # 统计不同错误之间的相似度
        different_error_line_similarities = []
        different_error_nac_similarities = []
        
        # 对每种错误类型，计算向量之间的相似度
        for error_signature, iteration_nums in error_groups.items():
            if len(iteration_nums) < 2:
                continue  # 只有一个用例，无法计算相似度
            
            vectors_data = all_error_vectors.get(error_signature, {})
            if len(vectors_data) < 2:
                continue
            
            print("=" * 80)
            print(f"错误类型: {error_signature}")
            print(f"触发该错误的迭代序号: {sorted([int(x) for x in iteration_nums])}")
            print(f"用例数量: {len(iteration_nums)}\n")
            
            f.write("=" * 100 + "\n")
            f.write(f"错误类型: {error_signature}\n")
            f.write(f"触发该错误的迭代序号: {sorted([int(x) for x in iteration_nums])}\n")
            f.write(f"用例数量: {len(iteration_nums)}\n\n")
            
            # 计算行覆盖向量的相似度
            print("--- 行覆盖向量 (line_vector) 相似度 ---")
            f.write("--- 行覆盖向量 (line_vector) 相似度 ---\n")
            
            line_similarities, line_lengths, line_invalid = calculate_similarities_for_group(vectors_data, 'line_vector')
            
            # 检查所有向量长度是否一致
            unique_lengths = set(line_lengths.values())
            if len(unique_lengths) > 1:
                print(f"警告: 行覆盖向量长度不一致!")
                f.write(f"警告: 行覆盖向量长度不一致!\n")
                for iter_num, length in line_lengths.items():
                    print(f"  迭代 {iter_num}: 长度 {length}")
                    f.write(f"  迭代 {iter_num}: 长度 {length}\n")
            else:
                print(f"所有行覆盖向量长度一致: {list(unique_lengths)[0]}")
                f.write(f"所有行覆盖向量长度一致: {list(unique_lengths)[0]}\n")
            
            # 输出相似度结果
            for (iter1, iter2) in combinations(sorted(vectors_data.keys(), key=int), 2):
                vec1 = vectors_data[iter1]['line_vector']
                vec2 = vectors_data[iter2]['line_vector']
                similarity, error_msg = cosine_similarity(vec1, vec2)
                
                if error_msg:
                    print(f"  迭代 {iter1} vs {iter2}: {error_msg}")
                    f.write(f"  迭代 {iter1} vs {iter2}: {error_msg}\n")
                else:
                    print(f"  迭代 {iter1} vs {iter2}: {similarity:.6f}")
                    f.write(f"  迭代 {iter1} vs {iter2}: {similarity:.6f}\n")
            
            if line_similarities:
                avg_sim = sum(line_similarities) / len(line_similarities)
                min_sim = min(line_similarities)
                max_sim = max(line_similarities)
                print(f"  平均相似度: {avg_sim:.6f}")
                print(f"  最小相似度: {min_sim:.6f}")
                print(f"  最大相似度: {max_sim:.6f}")
                f.write(f"  平均相似度: {avg_sim:.6f}\n")
                f.write(f"  最小相似度: {min_sim:.6f}\n")
                f.write(f"  最大相似度: {max_sim:.6f}\n")
                same_error_line_similarities.extend(line_similarities)
            
            print()
            f.write("\n")
            
            # 计算神经元覆盖向量的相似度
            print("--- 神经元覆盖向量 (nac_vector) 相似度 ---")
            f.write("--- 神经元覆盖向量 (nac_vector) 相似度 ---\n")
            
            nac_similarities, nac_lengths, nac_invalid = calculate_similarities_for_group(vectors_data, 'nac_vector')
            
            # 检查所有向量长度是否一致
            unique_lengths = set(nac_lengths.values())
            if len(unique_lengths) > 1:
                print(f"警告: 神经元覆盖向量长度不一致!")
                f.write(f"警告: 神经元覆盖向量长度不一致!\n")
                for iter_num, length in nac_lengths.items():
                    print(f"  迭代 {iter_num}: 长度 {length}")
                    f.write(f"  迭代 {iter_num}: 长度 {length}\n")
            else:
                print(f"所有神经元覆盖向量长度一致: {list(unique_lengths)[0]}")
                f.write(f"所有神经元覆盖向量长度一致: {list(unique_lengths)[0]}\n")
            
            # 输出相似度结果
            for (iter1, iter2) in combinations(sorted(vectors_data.keys(), key=int), 2):
                vec1 = vectors_data[iter1]['nac_vector']
                vec2 = vectors_data[iter2]['nac_vector']
                similarity, error_msg = cosine_similarity(vec1, vec2)
                
                if error_msg:
                    print(f"  迭代 {iter1} vs {iter2}: {error_msg}")
                    f.write(f"  迭代 {iter1} vs {iter2}: {error_msg}\n")
                else:
                    print(f"  迭代 {iter1} vs {iter2}: {similarity:.6f}")
                    f.write(f"  迭代 {iter1} vs {iter2}: {similarity:.6f}\n")
            
            if nac_similarities:
                avg_sim = sum(nac_similarities) / len(nac_similarities)
                min_sim = min(nac_similarities)
                max_sim = max(nac_similarities)
                print(f"  平均相似度: {avg_sim:.6f}")
                print(f"  最小相似度: {min_sim:.6f}")
                print(f"  最大相似度: {max_sim:.6f}")
                f.write(f"  平均相似度: {avg_sim:.6f}\n")
                f.write(f"  最小相似度: {min_sim:.6f}\n")
                f.write(f"  最大相似度: {max_sim:.6f}\n")
                same_error_nac_similarities.extend(nac_similarities)
            
            print("\n")
            f.write("\n\n")
        
        # 计算不同错误类型之间的相似度
        print("\n" + "=" * 80)
        print("计算不同错误类型之间的向量相似度")
        print("=" * 80)
        f.write("\n" + "=" * 100 + "\n")
        f.write("不同错误类型之间的向量相似度\n")
        f.write("=" * 100 + "\n\n")
        
        error_signatures = list(all_error_vectors.keys())
        for i, error1 in enumerate(error_signatures):
            for error2 in error_signatures[i+1:]:
                vectors1 = all_error_vectors[error1]
                vectors2 = all_error_vectors[error2]
                
                print(f"\n错误类型1: {error1}")
                print(f"  迭代序号: {sorted([int(x) for x in vectors1.keys()], key=int)}")
                print(f"错误类型2: {error2}")
                print(f"  迭代序号: {sorted([int(x) for x in vectors2.keys()], key=int)}")
                
                f.write(f"\n错误类型1: {error1}\n")
                f.write(f"  迭代序号: {sorted([int(x) for x in vectors1.keys()], key=int)}\n")
                f.write(f"错误类型2: {error2}\n")
                f.write(f"  迭代序号: {sorted([int(x) for x in vectors2.keys()], key=int)}\n")
                
                # 计算行覆盖向量相似度
                print("--- 行覆盖向量 (line_vector) 相似度 ---")
                f.write("--- 行覆盖向量 (line_vector) 相似度 ---\n")
                
                line_sims = []
                for iter1 in vectors1.keys():
                    for iter2 in vectors2.keys():
                        vec1 = vectors1[iter1]['line_vector']
                        vec2 = vectors2[iter2]['line_vector']
                        similarity, error_msg = cosine_similarity(vec1, vec2)
                        
                        if error_msg:
                            print(f"  迭代 {iter1} vs {iter2}: {error_msg}")
                            f.write(f"  迭代 {iter1} vs {iter2}: {error_msg}\n")
                        else:
                            print(f"  迭代 {iter1} vs {iter2}: {similarity:.6f}")
                            f.write(f"  迭代 {iter1} vs {iter2}: {similarity:.6f}\n")
                            line_sims.append(similarity)
                
                if line_sims:
                    avg_sim = sum(line_sims) / len(line_sims)
                    min_sim = min(line_sims)
                    max_sim = max(line_sims)
                    print(f"  平均相似度: {avg_sim:.6f}")
                    print(f"  最小相似度: {min_sim:.6f}")
                    print(f"  最大相似度: {max_sim:.6f}")
                    f.write(f"  平均相似度: {avg_sim:.6f}\n")
                    f.write(f"  最小相似度: {min_sim:.6f}\n")
                    f.write(f"  最大相似度: {max_sim:.6f}\n")
                    different_error_line_similarities.extend(line_sims)
                
                print()
                f.write("\n")
                
                # 计算神经元覆盖向量相似度
                print("--- 神经元覆盖向量 (nac_vector) 相似度 ---")
                f.write("--- 神经元覆盖向量 (nac_vector) 相似度 ---\n")
                
                nac_sims = []
                for iter1 in vectors1.keys():
                    for iter2 in vectors2.keys():
                        vec1 = vectors1[iter1]['nac_vector']
                        vec2 = vectors2[iter2]['nac_vector']
                        similarity, error_msg = cosine_similarity(vec1, vec2)
                        
                        if error_msg:
                            print(f"  迭代 {iter1} vs {iter2}: {error_msg}")
                            f.write(f"  迭代 {iter1} vs {iter2}: {error_msg}\n")
                        else:
                            print(f"  迭代 {iter1} vs {iter2}: {similarity:.6f}")
                            f.write(f"  迭代 {iter1} vs {iter2}: {similarity:.6f}\n")
                            nac_sims.append(similarity)
                
                if nac_sims:
                    avg_sim = sum(nac_sims) / len(nac_sims)
                    min_sim = min(nac_sims)
                    max_sim = max(nac_sims)
                    print(f"  平均相似度: {avg_sim:.6f}")
                    print(f"  最小相似度: {min_sim:.6f}")
                    print(f"  最大相似度: {max_sim:.6f}")
                    f.write(f"  平均相似度: {avg_sim:.6f}\n")
                    f.write(f"  最小相似度: {min_sim:.6f}\n")
                    f.write(f"  最大相似度: {max_sim:.6f}\n")
                    different_error_nac_similarities.extend(nac_sims)
                
                print("\n")
                f.write("\n")
        
        # 对比统计
        print("\n" + "=" * 80)
        print("相似度对比统计")
        print("=" * 80)
        f.write("\n" + "=" * 100 + "\n")
        f.write("相似度对比统计\n")
        f.write("=" * 100 + "\n\n")
        
        if same_error_line_similarities and different_error_line_similarities:
            print("\n行覆盖向量 (line_vector) 对比:")
            print(f"  相同错误内部:")
            print(f"    平均相似度: {sum(same_error_line_similarities) / len(same_error_line_similarities):.6f}")
            print(f"    最小相似度: {min(same_error_line_similarities):.6f}")
            print(f"    最大相似度: {max(same_error_line_similarities):.6f}")
            print(f"    样本数量: {len(same_error_line_similarities)}")
            print(f"  不同错误之间:")
            print(f"    平均相似度: {sum(different_error_line_similarities) / len(different_error_line_similarities):.6f}")
            print(f"    最小相似度: {min(different_error_line_similarities):.6f}")
            print(f"    最大相似度: {max(different_error_line_similarities):.6f}")
            print(f"    样本数量: {len(different_error_line_similarities)}")
            
            f.write("\n行覆盖向量 (line_vector) 对比:\n")
            f.write(f"  相同错误内部:\n")
            f.write(f"    平均相似度: {sum(same_error_line_similarities) / len(same_error_line_similarities):.6f}\n")
            f.write(f"    最小相似度: {min(same_error_line_similarities):.6f}\n")
            f.write(f"    最大相似度: {max(same_error_line_similarities):.6f}\n")
            f.write(f"    样本数量: {len(same_error_line_similarities)}\n")
            f.write(f"  不同错误之间:\n")
            f.write(f"    平均相似度: {sum(different_error_line_similarities) / len(different_error_line_similarities):.6f}\n")
            f.write(f"    最小相似度: {min(different_error_line_similarities):.6f}\n")
            f.write(f"    最大相似度: {max(different_error_line_similarities):.6f}\n")
            f.write(f"    样本数量: {len(different_error_line_similarities)}\n")
        
        if same_error_nac_similarities and different_error_nac_similarities:
            print("\n神经元覆盖向量 (nac_vector) 对比:")
            print(f"  相同错误内部:")
            print(f"    平均相似度: {sum(same_error_nac_similarities) / len(same_error_nac_similarities):.6f}")
            print(f"    最小相似度: {min(same_error_nac_similarities):.6f}")
            print(f"    最大相似度: {max(same_error_nac_similarities):.6f}")
            print(f"    样本数量: {len(same_error_nac_similarities)}")
            print(f"  不同错误之间:")
            print(f"    平均相似度: {sum(different_error_nac_similarities) / len(different_error_nac_similarities):.6f}")
            print(f"    最小相似度: {min(different_error_nac_similarities):.6f}")
            print(f"    最大相似度: {max(different_error_nac_similarities):.6f}")
            print(f"    样本数量: {len(different_error_nac_similarities)}")
            
            f.write("\n神经元覆盖向量 (nac_vector) 对比:\n")
            f.write(f"  相同错误内部:\n")
            f.write(f"    平均相似度: {sum(same_error_nac_similarities) / len(same_error_nac_similarities):.6f}\n")
            f.write(f"    最小相似度: {min(same_error_nac_similarities):.6f}\n")
            f.write(f"    最大相似度: {max(same_error_nac_similarities):.6f}\n")
            f.write(f"    样本数量: {len(same_error_nac_similarities)}\n")
            f.write(f"  不同错误之间:\n")
            f.write(f"    平均相似度: {sum(different_error_nac_similarities) / len(different_error_nac_similarities):.6f}\n")
            f.write(f"    最小相似度: {min(different_error_nac_similarities):.6f}\n")
            f.write(f"    最大相似度: {max(different_error_nac_similarities):.6f}\n")
            f.write(f"    样本数量: {len(different_error_nac_similarities)}\n")
        
        print(f"\n结果已保存到: {output_file}")

if __name__ == '__main__':
    main()
