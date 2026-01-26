# 对每个迭代的覆盖向量按照模块拼接，将拼接后的几个模块覆盖向量 追加写入npy文件中，方便后面的计算。
# 遍历/media/lzq/D/lzq/pylot_test/pylot/cov_vector下的以array_vector.npy结尾的文件(即{i}_array_vector.npy)，读取每个npy文件中的所有覆盖向量，按照模块拼接，追加写入对应的npy文件中。
# 模块定义如下，如果某个npy文件缺少某些对应名称的数组，则用0填充对应长度。

'''
模块定义：
所有模块total_vector：所有文件的覆盖向量拼接的覆盖向量。
感知模块（perception_vector，所有名称以pylot/perception开头的数组拼接的数组），包括：
   
  数组名称: pylot/perception/detection/obstacle.py_vector
    数组形状: (342,)
    总行数: 341, 覆盖: 24, 未覆盖: 118, 空行/注释: 199

  数组名称: pylot/perception/detection/speed_limit_sign.py_vector
    数组形状: (76,)
    总行数: 75, 覆盖: 2, 未覆盖: 25, 空行/注释: 48

  数组名称: pylot/perception/detection/traffic_light.py_vector
    数组形状: (424,)
    总行数: 423, 覆盖: 38, 未覆盖: 110, 空行/注释: 275

  数组名称: pylot/perception/detection/utils.py_vector
    数组形状: (595,)
    总行数: 594, 覆盖: 74, 未覆盖: 187, 空行/注释: 333

  数组名称: pylot/perception/point_cloud.py_vector
    数组形状: (213,)
    总行数: 212, 覆盖: 21, 未覆盖: 67, 空行/注释: 124

  数组名称: pylot/perception/detection/detection_operator.py_vector
    数组形状: (215,)
    总行数: 214, 覆盖: 34, 未覆盖: 58, 空行/注释: 122

  数组名称: pylot/perception/segmentation/segmented_frame.py_vector
    数组形状: (323,)
    总行数: 322, 覆盖: 12, 未覆盖: 119, 空行/注释: 191

  数组名称: pylot/perception/tracking/multi_object_tracker.py_vector
    数组形状: (21,)
    总行数: 20, 覆盖: 4, 未覆盖: 3, 空行/注释: 13

  数组名称: pylot/perception/tracking/object_tracker_operator.py_vector
    数组形状: (161,)
    总行数: 160, 覆盖: 55, 未覆盖: 51, 空行/注释: 54

  数组名称: pylot/perception/tracking/obstacle_location_history_operator.py_vector
    数组形状: (135,)
    总行数: 134, 覆盖: 50, 未覆盖: 43, 空行/注释: 41

  数组名称: pylot/perception/tracking/obstacle_trajectory.py_vector
    数组形状: (89,)
    总行数: 88, 覆盖: 8, 未覆盖: 40, 空行/注释: 40

  数组名称: pylot/perception/tracking/sort_tracker.py_vector
    数组形状: (70,)
    总行数: 69, 覆盖: 34, 未覆盖: 1, 空行/注释: 34

  数组名称: pylot/perception/detection/lane.py_vector
    数组形状: (213,)
    总行数: 212, 覆盖: 29, 未覆盖: 109, 空行/注释: 74

规划模块（planning_vector，所有名称以pylot/planning开头的数组拼接的数组），包括：

  数组名称: pylot/planning/behavior_planning_operator.py_vector
    数组形状: (277,)
    总行数: 276, 覆盖: 35, 未覆盖: 104, 空行/注释: 137

  数组名称: pylot/planning/planner.py_vector
    数组形状: (44,)
    总行数: 43, 覆盖: 23, 未覆盖: 2, 空行/注释: 18

  数组名称: pylot/planning/planning_operator.py_vector
    数组形状: (294,)
    总行数: 293, 覆盖: 56, 未覆盖: 98, 空行/注释: 139

  数组名称: pylot/planning/waypoints.py_vector
    数组形状: (218,)
    总行数: 217, 覆盖: 27, 未覆盖: 99, 空行/注释: 91

  数组名称: pylot/planning/world.py_vector
    数组形状: (444,)
    总行数: 443, 覆盖: 34, 未覆盖: 180, 空行/注释: 229

  数组名称: pylot/planning/rrt_star/rrt_star_planner.py_vector
    数组形状: (108,)
    总行数: 107, 覆盖: 25, 未覆盖: 14, 空行/注释: 68


控制模块（control_vector，所有名称以pylot/control开头的数组拼接的数组），包括：

  数组名称: pylot/control/mpc/mpc_operator.py_vector
    数组形状: (157,)
    总行数: 156, 覆盖: 53, 未覆盖: 60, 空行/注释: 43

  数组名称: pylot/control/mpc/mpc.py_vector
    数组形状: (324,)
    总行数: 323, 覆盖: 17, 未覆盖: 127, 空行/注释: 179

  数组名称: pylot/control/pid.py_vector
    数组形状: (129,)
    总行数: 128, 覆盖: 7, 未覆盖: 51, 空行/注释: 70

  数组名称: pylot/control/time_to_decision_operator.py_vector
    数组形状: (52,)
    总行数: 51, 覆盖: 10, 未覆盖: 24, 空行/注释: 17

  数组名称: pylot/control/mpc/utils.py_vector
    数组形状: (393,)
    总行数: 392, 覆盖: 109, 未覆盖: 72, 空行/注释: 211


其他模块（other_vector，所有名称不以pylot/perception、pylot/planning、pylot/control开头的数组拼接的数组）。



'''

import os
import numpy as np
from collections import defaultdict


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


def get_all_array_names_and_lengths(vectors_dir, exclude_keyword='pythonfuzz'):
    """
    扫描所有npy文件，获取所有数组名称及其最大长度
    
    返回:
        array_max_lengths: {array_name: max_length} 字典
    """
    array_max_lengths = {}
    
    if not os.path.exists(vectors_dir):
        return array_max_lengths
    
    for filename in os.listdir(vectors_dir):
        if filename.endswith('_array_vector.npy'):
            file_path = os.path.join(vectors_dir, filename)
            data = read_npy_file(file_path)
            
            if data and isinstance(data, dict):
                for key, value in data.items():
                    # 排除包含exclude_keyword的数组
                    if exclude_keyword not in key:
                        if isinstance(value, np.ndarray):
                            current_length = len(value)
                            if key not in array_max_lengths:
                                array_max_lengths[key] = current_length
                            else:
                                array_max_lengths[key] = max(array_max_lengths[key], current_length)
    
    return array_max_lengths


def classify_arrays_by_module(array_names, exclude_keyword='pythonfuzz'):
    """
    将数组名称按照模块分类，排除包含指定关键字的数组
    
    返回:
        perception_arrays: 感知模块的数组名称列表（排序）
        planning_arrays: 规划模块的数组名称列表（排序）
        control_arrays: 控制模块的数组名称列表（排序）
        other_arrays: 其他模块的数组名称列表（排序，排除pythonfuzz）
    """
    perception_arrays = []
    planning_arrays = []
    control_arrays = []
    other_arrays = []
    
    for array_name in array_names:
        # 排除包含exclude_keyword的数组
        if exclude_keyword in array_name:
            continue
        
        if array_name.startswith('pylot/perception'):
            perception_arrays.append(array_name)
        elif array_name.startswith('pylot/planning'):
            planning_arrays.append(array_name)
        elif array_name.startswith('pylot/control'):
            control_arrays.append(array_name)
        else:
            other_arrays.append(array_name)
    
    # 排序以确保顺序一致
    perception_arrays.sort()
    planning_arrays.sort()
    control_arrays.sort()
    other_arrays.sort()
    
    return perception_arrays, planning_arrays, control_arrays, other_arrays


def get_array_vector(data_dict, array_name, max_length, default_dtype=np.int32, verbose=False):
    """
    从字典中获取数组，如果缺失或长度不足，用0填充
    
    参数:
        data_dict: 数据字典
        array_name: 数组名称
        max_length: 最大长度
        default_dtype: 默认数据类型
        verbose: 是否打印详细信息
    
    返回:
        array: numpy数组
    """
    if array_name in data_dict and isinstance(data_dict[array_name], np.ndarray):
        vec = data_dict[array_name]
        if len(vec) < max_length:
            if verbose:
                print(f"数组 {array_name} 长度不足 ({len(vec)} < {max_length})，用0填充到最大长度")
            # 用0填充到最大长度
            padding = np.zeros(max_length - len(vec), dtype=vec.dtype)
            vec = np.concatenate([vec, padding])
        return vec
    else:
        if verbose:
            print(f"数组 {array_name} 缺失，用0填充到最大长度 {max_length}")
        # 缺失的数组用0填充（在对应位置补0）
        return np.zeros(max_length, dtype=default_dtype)


def splice_module_vectors(data_dict, array_max_lengths, 
                          standard_perception_arrays, standard_planning_arrays,
                          standard_control_arrays, standard_other_arrays,
                          exclude_keyword='pythonfuzz'):
    """
    按照模块拼接覆盖向量，确保所有文件使用相同的数组顺序和数量
    
    参数:
        data_dict: 当前文件的数据字典
        array_max_lengths: 所有数组的最大长度字典
        standard_perception_arrays: 标准感知模块数组列表（固定顺序）
        standard_planning_arrays: 标准规划模块数组列表（固定顺序）
        standard_control_arrays: 标准控制模块数组列表（固定顺序）
        standard_other_arrays: 标准其他模块数组列表（固定顺序）
        exclude_keyword: 要排除的关键字
    
    返回:
        total_vector: 所有数组拼接的向量
        perception_vector: 感知模块向量
        planning_vector: 规划模块向量
        control_vector: 控制模块向量
        other_vector: 其他模块向量
    """
    # 拼接各个模块的向量（按照标准顺序，缺失的用0填充）
    perception_vectors = []
    planning_vectors = []
    control_vectors = []
    other_vectors = []
    
    # 处理感知模块（按照标准顺序，缺失的数组在对应位置用0填充）
    for array_name in standard_perception_arrays:
        vec = get_array_vector(data_dict, array_name, array_max_lengths[array_name], verbose=False)
        perception_vectors.append(vec)
    
    # 处理规划模块（按照标准顺序，缺失的数组在对应位置用0填充）
    for array_name in standard_planning_arrays:
        vec = get_array_vector(data_dict, array_name, array_max_lengths[array_name], verbose=False)
        planning_vectors.append(vec)
    
    # 处理控制模块（按照标准顺序，缺失的数组在对应位置用0填充）
    for array_name in standard_control_arrays:
        vec = get_array_vector(data_dict, array_name, array_max_lengths[array_name], verbose=False)
        control_vectors.append(vec)
    
    # 处理其他模块（按照标准顺序，缺失的数组在对应位置用0填充）
    for array_name in standard_other_arrays:
        vec = get_array_vector(data_dict, array_name, array_max_lengths[array_name], verbose=False)
        other_vectors.append(vec)
    
    # 拼接所有模块向量（按照固定顺序：perception -> planning -> control -> other）
    all_vectors = perception_vectors + planning_vectors + control_vectors + other_vectors
    
    # 拼接所有向量
    total_vector = np.concatenate(all_vectors) if all_vectors else np.array([], dtype=np.int32)
    perception_vector = np.concatenate(perception_vectors) if perception_vectors else np.array([], dtype=np.int32)
    planning_vector = np.concatenate(planning_vectors) if planning_vectors else np.array([], dtype=np.int32)
    control_vector = np.concatenate(control_vectors) if control_vectors else np.array([], dtype=np.int32)
    other_vector = np.concatenate(other_vectors) if other_vectors else np.array([], dtype=np.int32)
    
    return total_vector, perception_vector, planning_vector, control_vector, other_vector


def main():
    vectors_dir = '/media/lzq/D/lzq/pylot_test/pylot/ori_cov_vector'
    
    print("=" * 80)
    print("开始处理覆盖向量模块拼接")
    print("=" * 80)
    
    # 第一步：扫描所有文件，获取所有数组名称及其最大长度
    print("\n第一步：扫描所有文件，确定数组标准长度和顺序...")
    array_max_lengths = get_all_array_names_and_lengths(vectors_dir)
    print(f"找到 {len(array_max_lengths)} 个不同的数组")
    
    # 确定每个模块的标准数组列表（固定顺序，所有文件都使用这个顺序）
    filtered_array_names = [name for name in array_max_lengths.keys() 
                           if 'pythonfuzz' not in name]
    standard_perception_arrays, standard_planning_arrays, standard_control_arrays, standard_other_arrays = \
        classify_arrays_by_module(filtered_array_names)
    
    print(f"\n标准数组列表（所有文件将按照此顺序拼接）:")
    print(f"  感知模块: {len(standard_perception_arrays)} 个数组")
    print(f"  规划模块: {len(standard_planning_arrays)} 个数组")
    print(f"  控制模块: {len(standard_control_arrays)} 个数组")
    print(f"  其他模块: {len(standard_other_arrays)} 个数组")
    
    # 第二步：处理每个文件
    print("\n第二步：处理每个文件，拼接模块向量...")
    
    processed_count = 0
    error_count = 0
    
    if os.path.exists(vectors_dir):
        for filename in sorted(os.listdir(vectors_dir)):
            if filename.endswith('_array_vector.npy'):
                file_path = os.path.join(vectors_dir, filename)
                
                # 读取文件
                data = read_npy_file(file_path)
                
                if not data or not isinstance(data, dict):
                    print(f"  跳过 {filename}: 无法读取或格式不正确")
                    error_count += 1
                    continue
                
                # 拼接模块向量（使用标准数组列表，确保顺序和数量一致）
                total_vec, perception_vec, planning_vec, control_vec, other_vec = \
                    splice_module_vectors(data, array_max_lengths,
                                        standard_perception_arrays, standard_planning_arrays,
                                        standard_control_arrays, standard_other_arrays)
                
                # 将模块向量追加到原字典中
                data['total_vector'] = total_vec
                data['perception_vector'] = perception_vec
                data['planning_vector'] = planning_vec
                data['control_vector'] = control_vec
                data['other_vector'] = other_vec
                
                # 保存回文件
                try:
                    np.save(file_path, data)
                    processed_count += 1
                    if processed_count % 10 == 0:
                        print(f"  已处理 {processed_count} 个文件...")
                except Exception as e:
                    print(f"  保存文件 {filename} 时出错: {e}")
                    error_count += 1
    
    print("\n" + "=" * 80)
    print("处理完成！")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    print("=" * 80)
    
    # 打印统计信息
    if array_max_lengths:
        print(f"\n最终模块统计:")
        print(f"  感知模块数组数: {len(standard_perception_arrays)}")
        print(f"  规划模块数组数: {len(standard_planning_arrays)}")
        print(f"  控制模块数组数: {len(standard_control_arrays)}")
        print(f"  其他模块数组数: {len(standard_other_arrays)}")
        print(f"  总数组数: {len(array_max_lengths)}")
        print(f"\n注意：所有文件的模块向量都包含相同数量和顺序的数组，缺失的数组已用0填充")


if __name__ == '__main__':
    main()
