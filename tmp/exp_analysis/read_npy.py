import os
import numpy as np

def read_npy_file(file_path):
    """
    读取npy文件并返回字典
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # 如果保存的是字典（通过np.save保存的字典），需要调用.item()
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        
        return data
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None


def print_dict_content(dict_data, file_name, max_items=100, max_array_elements=20):
    """
    打印字典的内容
    """
    if not isinstance(dict_data, dict):
        print(f"  {file_name}: 不是字典格式")
        return
    
    print(f"\n文件: {file_name}")
    print(f"包含 {len(dict_data)} 个数组")
    print("-" * 80)
    
    # 打印每个字典项（数组）
    for idx, (key, value) in enumerate(dict_data.items()):
        if idx >= max_items:
            print(f"\n  ... (还有 {len(dict_data) - max_items} 个数组未显示)")
            break
        
        print(f"\n  数组名称: {key}")
        
        if isinstance(value, np.ndarray):
            # print(f"    数组形状: {value.shape}")
            # print(f"    数据类型: {value.dtype}")
            
            # 统计覆盖情况（排除索引0，因为行号从1开始）
            if len(value) > 1:
                covered_lines = np.sum(value[1:] == 1)  # 覆盖的行数
                missed_lines = np.sum(value[1:] == 0)  # 未覆盖的行数
                empty_lines = np.sum(value[1:] == -1)  # 空行/注释的行数
                total_lines = len(value) - 1  # 总行数（排除索引0）
                
                # print(f"    总行数: {total_lines}, 覆盖: {covered_lines}, 未覆盖: {missed_lines}, 空行/注释: {empty_lines}")
            
            # 打印数组的部分内容
            if len(value) > 0:
                # 打印前几个非零元素（排除索引0）
                if len(value) > 1:
                    meaningful_indices = np.where(value[1:] != -1)[0] + 1  # +1因为从索引1开始
                    # if len(meaningful_indices) > 0:
                    #     sample_size = min(max_array_elements, len(meaningful_indices))
                    #     sample_indices = meaningful_indices[:sample_size]
                    #     sample_values = value[sample_indices]
                    #     print(f"    示例内容 (前{sample_size}个有意义的行):")
                    #     print(f"      行号: {sample_indices.tolist()}")
                    #     print(f"      值:   {sample_values.tolist()}")
                    # else:
                    #     print(f"    数组内容: {value[:min(10, len(value))].tolist()} (前10个元素)")
                else:
                    print(f"    数组内容: {value.tolist()}")
        else:
            print(f"    值类型: {type(value)}")
            print(f"    值: {value}")


def print_module_vectors_length(dict_data, file_name):
    """
    打印模块向量的长度信息
    """
    if not isinstance(dict_data, dict):
        print(f"  {file_name}: 不是字典格式")
        return
    
    print(f"\n文件: {file_name}")
    print("=" * 80)
    print("模块向量长度信息:")
    print("-" * 80)
    
    # 定义模块向量名称
    module_names = ['total_vector', 'perception_vector', 'planning_vector', 
                    'control_vector', 'other_vector']
    
    for module_name in module_names:
        if module_name in dict_data:
            vec = dict_data[module_name]
            if isinstance(vec, np.ndarray):
                print(f"  {module_name}:")
                print(f"    长度: {len(vec)}")
                print(f"    形状: {vec.shape}")
                print(f"    数据类型: {vec.dtype}")
            else:
                print(f"  {module_name}: 不是numpy数组 (类型: {type(vec)})")
        else:
            print(f"  {module_name}: 不存在")
    
    print("-" * 80)


def main():
    # 读取指定的文件
    file_path = "pylot/cov_vector/1_array_vector.npy"
    filename = "1_array_vector.npy"
    
    print("=" * 80)
    print(f"读取文件: {file_path}\n")
    
    # 读取文件
    dict_data = read_npy_file(file_path)
    
    if dict_data is not None:
        # 打印模块向量长度信息
        print_module_vectors_length(dict_data, filename)
        
        # 可选：打印所有内容（如果需要查看详细信息，可以取消注释）
        # print_dict_content(dict_data, filename)
        
        print("\n" + "=" * 80)
    else:
        print(f"无法读取文件: {filename}\n" + "=" * 80)


if __name__ == "__main__":
    main()
