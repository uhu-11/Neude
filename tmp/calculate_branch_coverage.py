#!/usr/bin/env python3
"""
计算status.json文件中所有py文件的分支覆盖率统计
"""

import json
import os
from pathlib import Path


def calculate_branch_coverage(json_file_path):
    """
    计算分支覆盖率统计
    
    Args:
        json_file_path: status.json文件的路径
        
    Returns:
        dict: 包含统计结果的字典
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化统计变量
    total_branches = 0
    total_partial_branches = 0
    total_missing_branches = 0
    py_file_count = 0
    
    # 遍历files中的所有文件
    files = data.get('files', {})
    
    for file_key, file_data in files.items():
        # 检查是否是py文件（从key判断，通常以_py结尾）
        # 或者从index.file字段判断
        index = file_data.get('index', {})
        file_path = index.get('file', '')
        
        # 判断是否是py文件
        is_py_file = file_key.endswith('_py') or file_path.endswith('.py')
        
        # 排除文件名中包含'pythonfuzz'的文件
        if is_py_file and 'pythonfuzz' not in file_path.lower():
            nums = index.get('nums', {})
            n_branches = nums.get('n_branches', 0)
            n_partial_branches = nums.get('n_partial_branches', 0)
            n_missing_branches = nums.get('n_missing_branches', 0)
            
            # 累加统计
            total_branches += n_branches
            total_partial_branches += n_partial_branches
            total_missing_branches += n_missing_branches
            py_file_count += 1
    
    # 计算覆盖到的分支
    covered_branches = total_branches - total_partial_branches - total_missing_branches
    
    # 计算分支覆盖率
    branch_coverage = 0.0
    if total_branches > 0:
        branch_coverage = covered_branches / total_branches
    
    # 构建结果字典
    results = {
        'total_branches': total_branches,
        'total_partial_branches': total_partial_branches,
        'total_missing_branches': total_missing_branches,
        'covered_branches': covered_branches,
        'branch_coverage': branch_coverage,
        'branch_coverage_percentage': branch_coverage * 100,
        'py_file_count': py_file_count
    }
    
    return results


def save_results(results, output_file):
    """
    保存结果到文件
    
    Args:
        results: 统计结果字典
        output_file: 输出文件路径
    """
    # 保存为文本文件
    txt_file = output_file.replace('.json', '.txt') if output_file.endswith('.json') else output_file + '.txt'
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("分支覆盖率统计结果\n")
        f.write("=" * 60 + "\n\n")
        # f.write(f"统计的Python文件数量: {results['py_file_count']}\n\n")
        f.write(f"总分支数量 (total_branches): {results['total_branches']}\n")
        f.write(f"总部分覆盖数量 (total_partial_branches): {results['total_partial_branches']}\n")
        f.write(f"总未覆盖数量 (total_missing_branches): {results['total_missing_branches']}\n\n")
        f.write(f"覆盖到的分支数量: {results['covered_branches']}\n")
        f.write(f"分支覆盖率: {results['branch_coverage']:.6f} ({results['branch_coverage_percentage']:.2f}%)\n")
        # f.write("\n" + "=" * 60 + "\n")
        # f.write("计算公式:\n")
        # f.write("覆盖到的分支 = total_branches - total_partial_branches - total_missing_branches\n")
        # f.write("分支覆盖率 = 覆盖到的分支 / total_branches\n")
        f.write("=" * 60 + "\n")
    
    # 保存为JSON文件
    json_file = output_file.replace('.txt', '.json') if output_file.endswith('.txt') else output_file + '.json'
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到:")
    print(f"  - 文本文件: {txt_file}")
    print(f"  - JSON文件: {json_file}")


def main():
    # 输入文件路径
    exp_name = "exp37"
    # /media/lzq/D/lzq/pylot_test/pylot/covreport/total100
    # json_file_path = f"/media/lzq/My Passport/pylot实验数据/实验结果/{exp_name}/covhtml/totalM/status.json"
    # json_file_path = f"/media/lzq/My Passport/pylot实验数据/实验结果/{exp_name}/covhtml/total100/status.json"

    json_file_path = f"/media/lzq/D/lzq/pylot_test/pylot/covhtml/total100/status.json"
    
    # 输出文件路径（保存在当前工作目录）
    # output_file = f"branch_coverage_results_{exp_name}"
    output_file = f"branch_coverage_results_py_{exp_name}"
    
    print(f"正在读取文件: {json_file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误: 文件不存在: {json_file_path}")
        return
    
    # 计算分支覆盖率
    print("正在计算分支覆盖率统计...")
    results = calculate_branch_coverage(json_file_path)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("分支覆盖率统计结果")
    print("=" * 60)
    print(f"统计的Python文件数量: {results['py_file_count']}")
    print(f"\n总分支数量 (total_branches): {results['total_branches']}")
    print(f"总部分覆盖数量 (total_partial_branches): {results['total_partial_branches']}")
    print(f"总未覆盖数量 (total_missing_branches): {results['total_missing_branches']}")
    print(f"\n覆盖到的分支数量: {results['covered_branches']}")
    print(f"分支覆盖率: {results['branch_coverage']:.6f} ({results['branch_coverage_percentage']:.2f}%)")
    print("=" * 60 + "\n")
    
    # 保存结果
    save_results(results, output_file)


if __name__ == "__main__":
    main()

