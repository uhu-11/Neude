'''
我想查看'/media/lzq/My Passport/pylot实验数据/实验结果/exp16/error_seeds_vectors'中1.pickle、2.pickle、3.pickle文件中的nac_vector的长度。

还想查看'/media/lzq/My Passport/pylot实验数据/实验结果/exp1/error_seeds_vectors'中1.pickle、2.pickle、3.pickle文件中的nac_vector的长度。
'''
import pickle
import os
import sys

# 添加必要的路径以导入自定义模块
sys.path.insert(0, '/media/lzq/D/lzq/pylot_test')
sys.path.insert(0, '/media/lzq/D/lzq/pylot_test/pythonfuzz')

def check_nac_vector_length(pickle_path):
    """检查pickle文件中nac_vector的长度"""
    if not os.path.exists(pickle_path):
        print(f"  文件不存在: {pickle_path}")
        return None
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            if 'nac_vector' in data:
                nac_vector = data['nac_vector']
                length = len(nac_vector) if hasattr(nac_vector, '__len__') else None
                return length
            else:
                print(f"  文件中没有 'nac_vector' 键")
                print(f"  可用键: {list(data.keys())}")
                return None
        else:
            print(f"  文件不是字典格式，类型: {type(data)}")
            if hasattr(data, 'nac_vector'):
                nac_vector = getattr(data, 'nac_vector')
                length = len(nac_vector) if hasattr(nac_vector, '__len__') else None
                return length
            return None
    except Exception as e:
        print(f"  读取文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 定义要检查的路径和文件
    paths_to_check = [
        {
            'base_path': '/media/lzq/My Passport/pylot实验数据/实验结果/exp16/error_seeds_vectors',
            'files': ['1.pickle', '2.pickle', '3.pickle'],
            'label': 'exp16'
        },
        {
            'base_path': '/media/lzq/My Passport/pylot实验数据/实验结果/exp1/error_seeds_vectors',
            'files': ['1.pickle', '2.pickle', '3.pickle'],
            'label': 'exp1'
        },
        {
            'base_path': '/media/lzq/My Passport/pylot实验数据/实验结果/exp2/error_seeds_vectors',
            'files': ['1.pickle', '2.pickle', '3.pickle'],
            'label': 'exp2'
        },
        {
            'base_path': '/media/lzq/My Passport/pylot实验数据/实验结果/exp17/error_seeds_vectors',
            'files': ['1.pickle', '2.pickle', '3.pickle'],
            'label': 'exp17'
        },
        {
            'base_path': '/media/lzq/My Passport/pylot实验数据/实验结果/exp18/error_seeds_vectors',
            'files': ['1.pickle', '2.pickle', '3.pickle'],
            'label': 'exp18'
        },
        {
            'base_path': '/media/lzq/My Passport/pylot实验数据/实验结果/exp19/error_seeds_vectors',
            'files': ['1.pickle', '2.pickle', '3.pickle'],
            'label': 'exp19'
        },
        {
            'base_path': '/media/lzq/My Passport/pylot实验数据/实验结果/exp20/error_seeds_vectors',
            'files': ['1.pickle', '2.pickle', '3.pickle'],
            'label': 'exp20'
        },
        
    ]
    
    print("=" * 80)
    print("检查 nac_vector 长度")
    print("=" * 80)
    print()
    
    results = {}
    
    for path_info in paths_to_check:
        base_path = path_info['base_path']
        files = path_info['files']
        label = path_info['label']
        
        print(f"检查路径: {base_path}")
        print(f"实验标签: {label}")
        print("-" * 80)
        
        results[label] = {}
        
        for filename in files:
            pickle_path = os.path.join(base_path, filename)
            print(f"  文件: {filename}")
            
            length = check_nac_vector_length(pickle_path)
            
            if length is not None:
                print(f"    nac_vector 长度: {length}")
                results[label][filename] = length
            else:
                results[label][filename] = None
            
            print()
        
        print()
    
    # 输出汇总结果
    print("=" * 80)
    print("汇总结果")
    print("=" * 80)
    
    for label, file_results in results.items():
        print(f"\n{label}:")
        print(f"  {'文件名':<20} {'nac_vector长度':<20}")
        print(f"  {'-'*20} {'-'*20}")
        for filename, length in file_results.items():
            length_str = str(length) if length is not None else "N/A"
            print(f"  {filename:<20} {length_str:<20}")
    
    # 输出所有实验的对比表格
    print("\n" + "=" * 80)
    print("所有实验对比表格")
    print("=" * 80)
    
    # 获取所有实验标签并排序
    all_labels = sorted(results.keys(), key=lambda x: int(x.replace('exp', '')) if x.replace('exp', '').isdigit() else 999)
    
    # 创建表头
    header = f"{'文件名':<20}"
    for label in all_labels:
        header += f" {label:<15}"
    print(header)
    print("-" * (20 + 16 * len(all_labels)))
    
    # 输出每个文件的所有实验数据
    for filename in ['1.pickle', '2.pickle', '3.pickle']:
        row = f"{filename:<20}"
        for label in all_labels:
            length = results[label].get(filename)
            length_str = str(length) if length is not None else "N/A"
            row += f" {length_str:<15}"
        print(row)
    
    # 计算平均值（如果有数据）
    print("\n" + "-" * (20 + 16 * len(all_labels)))
    avg_row = f"{'平均值':<20}"
    for label in all_labels:
        lengths = [results[label].get(f) for f in ['1.pickle', '2.pickle', '3.pickle']]
        valid_lengths = [l for l in lengths if l is not None]
        if valid_lengths:
            avg = sum(valid_lengths) / len(valid_lengths)
            avg_row += f" {avg:<15.1f}"
        else:
            avg_row += f" {'N/A':<15}"
    print(avg_row)

if __name__ == "__main__":
    main()
