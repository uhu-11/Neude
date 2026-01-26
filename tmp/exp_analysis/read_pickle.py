
# 读取pickle文件

import pickle

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    file_path = '/media/lzq/D/lzq/pylot_test/pylot/error_seeds_vectors/100.pickle'
    data = read_pickle(file_path)
    
    # 统计向量的长度
    if 'line_vector' in data:
        line_vector_len = len(data['line_vector'])
        print(f"line_vector 长度: {line_vector_len}")
    
    if 'nac_vector' in data:
        nac_vector_len = len(data['nac_vector'])
        print(f"nac_vector 长度: {nac_vector_len}")