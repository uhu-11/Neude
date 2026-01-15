'''
我想统计所有codelfuzz1实验的MR错误数量。
codelfuzz1实验的包含6个配置，每次实验结果在当前文件夹下的define_mr_err_output_codelfuzz1_{i}.txt文件中。i从1到6。
统计[0, 0, 0]，[0, 0, 1]，[0, 1, 0]，[0, 1, 1]，[1, 0, 0]，[1, 0, 1]，[1, 1, 0]，[1, 1, 1]这八种类型各自的总数量。


'''
import os
import re

# 定义8种错误类型
error_types = [
    '[0, 0, 0]',
    '[0, 0, 1]',
    '[0, 1, 0]',
    '[0, 1, 1]',
    '[1, 0, 0]',
    '[1, 0, 1]',
    '[1, 1, 0]',
    '[1, 1, 1]'
]

def parse_file(file_path):
    """解析单个文件，提取8种错误类型的数量"""
    counts = {error_type: 0 for error_type in error_types}
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return counts
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 使用正则表达式匹配每一行的错误类型和数量
        # 格式: [0, 0, 0]            895             89.50%
        pattern = r'\[(\d+),\s*(\d+),\s*(\d+)\]\s+(\d+)\s+'
        
        matches = re.findall(pattern, content)
        
        for match in matches:
            error_type = f'[{match[0]}, {match[1]}, {match[2]}]'
            count = int(match[3])
            
            if error_type in counts:
                counts[error_type] = count
                
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    
    return counts

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 初始化总计数
    total_counts = {error_type: 0 for error_type in error_types}
    
    # 读取6个文件
    # print("=" * 80)
    # print("统计所有codelfuzz1实验的MR错误数量")
    # print("=" * 80)
    # print()
    
    file_counts = []
    
    for i in range(1, 7):
        file_path = os.path.join(current_dir, f'define_mr_err_output_codelfuzz1_{i}.txt')
        # print(f"正在读取文件 {i}: {file_path}")
        
        counts = parse_file(file_path)
        file_counts.append((i, counts))
        
        # 累加到总数
        for error_type in error_types:
            total_counts[error_type] += counts[error_type]
        
        # print(f"  文件 {i} 统计:")
        # for error_type in error_types:
        #     print(f"    {error_type}: {counts[error_type]}")
        # print()
    
    # 输出汇总结果
    print("=" * 80)
    print("汇总统计结果")
    print("=" * 80)
    print(f"{'错误类型':<20} {'数量':<15} {'百分比':<15}")
    print("-" * 50)
    
    total = sum(total_counts.values())
    
    for error_type in error_types:
        count = total_counts[error_type]
        percentage = (count / total * 100) if total > 0 else 0.0
        print(f"{error_type:<20} {count:<15} {percentage:.2f}%")
    
    print()
    print(f"总计: {total}")
    print()
    
    # 输出每个文件的详细统计
    print("=" * 80)
    print("各文件详细统计")
    print("=" * 80)
    print(f"{'文件':<10} " + " ".join([f"{et:<12}" for et in error_types]))
    print("-" * 120)
    
    for file_num, counts in file_counts:
        counts_str = " ".join([f"{counts[et]:<12}" for et in error_types])
        print(f"文件{file_num:<6} {counts_str}")

if __name__ == "__main__":
    main()
