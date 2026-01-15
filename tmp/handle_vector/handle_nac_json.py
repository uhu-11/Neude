'''
读取/home/lzq/test_reslut/会议论文实验结果/exp35/100.json，取出json文件中的batch_nac_vectors，ious列的数据，
保存到/home/lzq/result/datas/100_nac_vectors_exp35.json文件中。

'''

import json
import os

def main():
    # 输入文件路径
    input_json_path = '/home/lzq/test_reslut/会议论文实验结果/exp35/100.json'
    
    # 输出文件路径
    output_json_path = '/home/lzq/result/datas/100_nac_vectors_exp35.json'
    
    # 读取JSON文件
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取JSON文件: {input_json_path}")
    except Exception as e:
        print(f"读取JSON文件失败 {input_json_path}: {e}")
        return
    
    # 提取batch_nac_vectors和ious
    batch_nac_vectors = data.get('batch_nac_vectors', [])
    ious = data.get('ious', [])
    
    print(f"读取到 {len(batch_nac_vectors)} 次迭代的神经元向量")
    print(f"读取到 {len(ious)} 次迭代的IoU值")
    
    # 构建输出数据
    output_data = {
        'batch_nac_vectors': batch_nac_vectors,
        'ious': ious
    }
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 保存到新的JSON文件
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"成功保存数据到: {output_json_path}")
        print(f"保存了 {len(batch_nac_vectors)} 次迭代的batch_nac_vectors")
        print(f"保存了 {len(ious)} 次迭代的ious")
    except Exception as e:
        print(f"保存JSON文件失败 {output_json_path}: {e}")


if __name__ == "__main__":
    main()
