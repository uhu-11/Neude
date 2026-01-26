import os
import numpy as np

# /media/lzq/D/lzq/pylot_test/pylot/cov_vector目录下有若干npy文件，文件名为{i}_vector.npy。每个npy文件保存了若干字典，每个字典保存了一个html文件的覆盖向量。
# 将/media/lzq/D/lzq/pylot_test/pylot/cov_vector目录下的所有npy文件中的字典转换为数组，数组和字典的名称相同，数组内容为字典的值，保存为对应的npy文件，文件名为{i}_array_vector.npy，存在/media/lzq/D/lzq/pylot_test/pylot/cov_vector目录下。
# 打印转换后的数组内容。

def convert_dict_to_arrays(input_file, output_file):
    """
    将npy文件中的字典转换为数组格式
    """
    # 读取字典
    data = np.load(input_file, allow_pickle=True)
    
    # 如果保存的是字典（通过np.save保存的字典），需要调用.item()
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    
    if not isinstance(data, dict):
        print(f"警告: {input_file} 不是字典格式，跳过")
        return None
    
    # 将字典转换为数组字典，确保所有值都是numpy数组
    arrays_dict = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # 数组名称和字典的key相同，数组内容为字典的值
            arrays_dict[key] = value
        else:
            print(f"警告: {key} 的值不是numpy数组，跳过")
    
    if not arrays_dict:
        print(f"警告: {input_file} 中没有有效的数组，跳过")
        return None
    
    # 保存为.npy文件，使用字典格式保存多个命名数组
    # 虽然文件扩展名是.npy，但内容仍然是字典，这样可以保留数组名称
    np.save(output_file, arrays_dict)
    
    return arrays_dict


def main():
    input_dir = "/media/lzq/D/lzq/pylot_test/pylot/cov_vector"
    output_dir = "/media/lzq/D/lzq/pylot_test/pylot/cov_vector"

    # input_dir = "/media/lzq/D/lzq/pylot_test/pylot/ori_cov_vector"
    # output_dir = "/media/lzq/D/lzq/pylot_test/pylot/ori_cov_vector"
    
    # 获取所有{i}_vector.npy文件
    files = [f for f in os.listdir(input_dir) if f.endswith('_vector.npy') and not f.endswith('_array_vector.npy')]
    
    # 按文件名中的数字排序
    def get_number(filename):
        try:
            return int(filename.split('_')[0])
        except ValueError:
            return 0
    
    files.sort(key=get_number)
    
    print(f"找到 {len(files)} 个文件需要转换\n")
    
    for filename in files:
        # 提取文件编号
        file_number = filename.split('_')[0]
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, f"{file_number}_array_vector.npy")
        
        print(f"处理文件: {filename}")
        
        # 转换字典为数组
        arrays_dict = convert_dict_to_arrays(input_file, output_file)
        
        if arrays_dict:
            print(f"  已保存到: {output_file}")
            print(f"  包含 {len(arrays_dict)} 个数组\n")
            
            # 打印数组内容
            print("  数组内容:")
            for array_name, array in arrays_dict.items():
                # 统计覆盖情况（排除索引0，因为行号从1开始）
                if len(array) > 1:
                    covered_lines = np.sum(array[1:] == 1)  # 覆盖的行数
                    missed_lines = np.sum(array[1:] == 0)  # 未覆盖的行数
                    empty_lines = np.sum(array[1:] == -1)  # 空行/注释的行数
                    total_lines = len(array) - 1  # 总行数（排除索引0）
                    
                    print(f"    {array_name}:")
                    print(f"      总行数={total_lines}, 覆盖={covered_lines}, 未覆盖={missed_lines}, 空行/注释={empty_lines}")
                    print(f"      数组形状={array.shape}, 数据类型={array.dtype}")
                    
                    # 打印前20个有意义的行作为示例（排除-1）
                    meaningful_indices = np.where(array[1:] != -1)[0] + 1  # +1因为从索引1开始
                    if len(meaningful_indices) > 0:
                        sample_indices = meaningful_indices[:20]
                        sample_values = array[sample_indices]
                        print(f"      示例行号={sample_indices.tolist()}")
                        print(f"      示例值={sample_values.tolist()}")
                else:
                    print(f"    {array_name}: 长度为{len(array)}, 无有效行")
            print()
        else:
            print(f"  转换失败或文件为空\n")


if __name__ == "__main__":
    main()
