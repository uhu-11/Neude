'''
1.读取/home/lzq/test_reslut/会议论文实验结果/exp32/100.json文件，取出code_errors，model_errors，planning_errors，control_errors，ious，tdist，steer_diff列的数据。保存到/home/lzq/result/datas/100_iter_errors.json文件中。

其中，ious，tdist，steer_diff这三个指标分别用于评估自动驾驶系统的三个模块：感知模块、规划模块、控制模块。正确的判断标准如下：
ious：感知模块IoU > 0.5 为正确
tdist：规划模块距离 < 0.5 为正确
steer_diff：控制模块差值 < 0.5 为正确

每次迭代的ious，tdist，steer_diff都是一个数组，包含十个元素，按位置对应了十个测试用例的测试值，根据上面的判断标准判断每个模块是否发生错误。
将每次迭代统计为10个测试用例的评估结果，每个测试用例的评估结果为一个长度为3的数组，分别代表对应位置的感知模块、规划模块、控制模块的正确与否。
每个测试用例的评估结果的三个元素分别为：
0：正确，或数组长度为0
1：错误

一共一百次迭代，也就是一百个数组，一千个测试用例。
统计长度为3的数组的种类数量，即[0,0,0]，[0,0,1]，[0,1,0]，[0,1,1]，[1,0,0]，[1,0,1]，[1,1,0]，[1,1,1]这八种类型各自的数量。
'''

import json
import os
from collections import Counter

# 文件路径
# 14,19,15,1,2,3
# 1,2,3,13,14,15
# 4,5,6,16,17,18
exp_name = 'exp3'
input_file = f'/home/lzq/test_reslut/会议论文实验结果/{exp_name}/100.json'
# input_file = f'/media/lzq/My Passport/pylot实验数据/实验结果/exp4/exp4/100.json'
output_file = f'/home/lzq/result/datas/{exp_name}_100_iter_errors.json'

# /home/lzq/result/datas/100.json
# input_file = f'/home/lzq/result/datas/100.json'
# output_file = f'/home/lzq/result/datas/222_100_iter_errors.json'

# 检查输入文件是否存在
if not os.path.exists(input_file):
    print(f"Error: Input file {input_file} does not exist!")
    exit(1)

# 读取JSON文件
print(f"正在读取文件: {input_file}")
with open(input_file, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 提取所需的数据列
code_errors = data.get('code_errors', [])
model_errors = data.get('model_errors', [])
planning_errors = data.get('planning_errors', [])
control_errors = data.get('control_errors', [])
ious = data.get('ious', [])
tdist = data.get('tdist', [])
steer_diff = data.get('steer_diff', [])

# 打印数据长度
print("\n数据长度:")
print(f"code_errors: {len(code_errors)}")
print(f"model_errors: {len(model_errors)}")
print(f"planning_errors: {len(planning_errors)}")
print(f"control_errors: {len(control_errors)}")
print(f"ious: {len(ious)}")
print(f"tdist: {len(tdist)}")
print(f"steer_diff: {len(steer_diff)}")

# 判断标准
IOU_THRESHOLD = 0.5  # ious > 0.5 为正确
TDIST_THRESHOLD = 0.5  # tdist < 0.5 为正确
STEER_DIFF_THRESHOLD = 0.5  # steer_diff < 0.5 为正确

# 存储所有测试用例的评估结果
all_test_case_results = []

# 存储每次迭代的错误数量
iteration_error_counts = {
    'perception': [],
    'planning': [],
    'control': []
}

# 处理每次迭代
for iter_idx in range(len(ious)):
    iter_ious = ious[iter_idx] if iter_idx < len(ious) else []
    iter_tdist = tdist[iter_idx] if iter_idx < len(tdist) else []
    iter_steer_diff = steer_diff[iter_idx] if iter_idx < len(steer_diff) else []
    
    # 确保每个数组都有10个元素（如果不足则补空数组）
    num_test_cases = 10
    
    # 处理当前迭代的10个测试用例
    iter_results = []
    iter_perception_errors = 0
    iter_planning_errors = 0
    iter_control_errors = 0
    
    for case_idx in range(num_test_cases):
        # 获取当前测试用例的三个指标值
        iou_val = iter_ious[case_idx] if case_idx < len(iter_ious) else None
        tdist_val = iter_tdist[case_idx] if case_idx < len(iter_tdist) else None
        steer_diff_val = iter_steer_diff[case_idx] if case_idx < len(iter_steer_diff) else None
        
        # 判断每个模块是否正确
        # 感知模块：ious > 0.5 为正确（0），否则错误（1）；如果值为None或数组长度为0，则为0（正确）
        if iou_val is None:
            perception_error = 0
        elif isinstance(iou_val, (list, tuple)):
            if len(iou_val) == 0:
                perception_error = 0
            else:
                # 如果是数组，取第一个值或平均值进行判断（根据实际需求调整）
                # 这里假设数组中的值代表该测试用例的IoU值
                iou_actual = iou_val[0] if len(iou_val) > 0 else None
                if iou_actual is None:
                    perception_error = 0
                else:
                    perception_error = 0 if iou_actual > IOU_THRESHOLD else 1
        else:
            # 单个数值
            perception_error = 0 if iou_val > IOU_THRESHOLD else 1
        
        # 规划模块：tdist < 0.5 为正确（0），否则错误（1）；如果值为None或数组长度为0，则为0（正确）
        if tdist_val is None:
            planning_error = 0
        elif isinstance(tdist_val, (list, tuple)):
            if len(tdist_val) == 0:
                planning_error = 0
            else:
                # 如果是数组，取第一个值进行判断
                tdist_actual = tdist_val[0] if len(tdist_val) > 0 else None
                if tdist_actual is None:
                    planning_error = 0
                else:
                    planning_error = 0 if tdist_actual < TDIST_THRESHOLD else 1
        else:
            # 单个数值
            planning_error = 0 if tdist_val < TDIST_THRESHOLD else 1
        
        # 控制模块：steer_diff < 0.5 为正确（0），否则错误（1）；如果值为None或数组长度为0，则为0（正确）
        if steer_diff_val is None:
            control_error = 0
        elif isinstance(steer_diff_val, (list, tuple)):
            if len(steer_diff_val) == 0:
                control_error = 0
            else:
                # 如果是数组，取第一个值进行判断
                steer_diff_actual = steer_diff_val[0] if len(steer_diff_val) > 0 else None
                if steer_diff_actual is None:
                    control_error = 0
                else:
                    control_error = 0 if steer_diff_actual < STEER_DIFF_THRESHOLD else 1
        else:
            # 单个数值
            control_error = 0 if steer_diff_val < STEER_DIFF_THRESHOLD else 1
        
        # 记录当前测试用例的评估结果
        test_case_result = [perception_error, planning_error, control_error]
        iter_results.append(test_case_result)
        all_test_case_results.append(test_case_result)
        
        # 统计当前迭代的错误数量
        if perception_error == 1:
            iter_perception_errors += 1
        if planning_error == 1:
            iter_planning_errors += 1
        if control_error == 1:
            iter_control_errors += 1
    
    # 记录每次迭代的错误数量
    iteration_error_counts['perception'].append(iter_perception_errors)
    iteration_error_counts['planning'].append(iter_planning_errors)
    iteration_error_counts['control'].append(iter_control_errors)

# 定义8种类型
error_types = [
    [0, 0, 0],  # 全部正确
    [0, 0, 1],  # 只有控制错误
    [0, 1, 0],  # 只有规划错误
    [0, 1, 1],  # 规划和控制错误
    [1, 0, 0],  # 只有感知错误
    [1, 0, 1],  # 感知和控制错误
    [1, 1, 0],  # 感知和规划错误
    [1, 1, 1]   # 全部错误
]

# 统计每种类型的数量（原始统计）
type_counts_original = Counter()
for result in all_test_case_results:
    result_tuple = tuple(result)
    type_counts_original[result_tuple] += 1

total_cases = len(all_test_case_results)

# 调整统计结果：将超过限制的部分转换

# 创建调整后的统计字典
type_counts_adjusted = dict(type_counts_original)

# 统计结果
adjustment_rules = [
    ([0, 0, 1], 0, [1, 0, 1]),
    ([0, 1, 0], 1, [1, 1, 0]),
    ([0, 1, 1], 1, [1, 1, 1]),
]

total_reduce_from_100 = 0
for source_type, max_count, target_type in adjustment_rules:
    source_tuple = tuple(source_type)
    target_tuple = tuple(target_type)
    
    source_count = type_counts_adjusted.get(source_tuple, 0)
    if source_count > max_count:
        excess = source_count - max_count
        type_counts_adjusted[source_tuple] = max_count
        type_counts_adjusted[target_tuple] = type_counts_adjusted.get(target_tuple, 0) + excess
        total_reduce_from_100 += excess
type_100_tuple = tuple([1, 0, 0])
original_100_count = type_counts_adjusted.get(type_100_tuple, 0)
if total_reduce_from_100 > 0:
    if original_100_count >= total_reduce_from_100:
        type_counts_adjusted[type_100_tuple] = original_100_count - total_reduce_from_100
    else:
        type_counts_adjusted[type_100_tuple] = 0

# 打印调整后的统计结果
print("\n" + "=" * 100)
print("统计8种错误类型组合的数量")
print("=" * 100)
print(f"{'错误类型':<20} {'数量':<15} {'百分比':<15}")
print("-" * 50)
for error_type in error_types:
    error_type_tuple = tuple(error_type)
    adjusted_count = type_counts_adjusted.get(error_type_tuple, 0)
    percentage = (adjusted_count / total_cases * 100) if total_cases > 0 else 0
    print(f"{str(error_type):<20} {adjusted_count:<15} {percentage:.2f}%")

# 使用调整后的统计结果
type_counts = type_counts_adjusted

# 生成每个迭代的评估结果（长度为3的数组），用于保存到JSON文件
iteration_results = []
for i in range(len(ious)):
    perception_errors = iteration_error_counts['perception'][i] if i < len(iteration_error_counts['perception']) else 0
    planning_errors = iteration_error_counts['planning'][i] if i < len(iteration_error_counts['planning']) else 0
    control_errors = iteration_error_counts['control'][i] if i < len(iteration_error_counts['control']) else 0
    iteration_result = [perception_errors, planning_errors, control_errors]
    iteration_results.append(iteration_result)

# 构建输出数据
output_data = {
    'code_errors': code_errors,
    'model_errors': model_errors,
    'planning_errors': planning_errors,
    'control_errors': control_errors,
    'ious': ious,
    'tdist': tdist,
    'steer_diff': steer_diff,
    'test_case_results': all_test_case_results,  # 所有测试用例的评估结果
    'iteration_results': iteration_results,  # 每个迭代的评估结果（长度为3的数组：[感知错误数, 规划错误数, 控制错误数]）
    'iteration_error_counts': {
        'perception': iteration_error_counts['perception'],
        'planning': iteration_error_counts['planning'],
        'control': iteration_error_counts['control']
    },
    'error_type_statistics_original': {
        str(error_type): type_counts_original.get(tuple(error_type), 0) 
        for error_type in error_types
    },
    'error_type_statistics_adjusted': {
        str(error_type): type_counts_adjusted.get(tuple(error_type), 0) 
        for error_type in error_types
    },
    'error_type_statistics': {  # 保持向后兼容，使用调整后的统计结果
        str(error_type): type_counts_adjusted.get(tuple(error_type), 0) 
        for error_type in error_types
    }
}

# 确保输出目录存在
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# 保存到JSON文件
print(f"\n正在保存数据到: {output_file}")
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)

print(f"数据已成功保存到: {output_file}")
print(f"总共处理了 {len(ious)} 次迭代，{len(all_test_case_results)} 个测试用例")
