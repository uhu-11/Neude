import os
import math
import random
import struct
import hashlib
from PIL import Image

import numpy as np
import pickle
from . import dictionnary
from .Mutation import Mutation
from .Seed import Seed
from . import config
from .SeedStrategy import SeedStrategy
from .utils import ParamsType, RandomInitForType
from .utils.ConvertFolderToNpy import load_images_from_folder
try:
    from random import _randbelow
except ImportError:
    from random import _inst
    _randbelow = _inst._randbelow

INTERESTING8 = [-128, -1, 0, 1, 16, 32, 64, 100, 127]
INTERESTING16 = [0, 128, 255, 256, 512, 1000, 1024, 4096, 32767, 65535]
INTERESTING32 = [0, 1, 32768, 65535, 65536, 100663045, 2147483647, 4294967295]


#进行种子的初始化以及种子变异操作生成新的测试用例

class Corpus(object):
    # 这个dirs应该就是初始用例的文件
    def __init__(self, dirs=None, max_input_size=4096, dict_path=None, target=None):
        # self._inputs 是一个列表，用于存储从 dirs 中读取的文件内容
        self.target = target
        self._inputs = []
        self._dict = dictionnary.Dictionary(dict_path)
        self._max_input_size = max_input_size
        self._dirs = dirs if dirs else []
        for i, path in enumerate(dirs):
            if i == 0 and not os.path.exists(path):
                os.mkdir(path)
            '''
            读取 dirs 中的文件，并将文件内容创建Seed对象，写入 self._inputs 列表中
            '''
            if os.path.isfile(path):
                params_type = ParamsType.get_parameter_types(target)
                self._add_seed(path, params_type)
            else:
                for i in os.listdir(path):
                    fname = os.path.join(path, i)
                    if os.path.isfile(fname):
                        self._add_seed(fname)
        #### 如果没有指定初始化种子的文件，那么就随机初始化一个种子(这种情况下必须要求目标函数的参数已经标注类型)
        if not dirs:
            random_params = {}
            params_type = ParamsType.get_parameter_types(target)
            
            # 如果获取不到参数类型（比如只有**kwargs的情况）
            if params_type is None:
                return
                
            # 为每个参数生成随机值
            for param_name, param_type in params_type.items():
                if param_type is None:
                    # 如果参数没有类型注解，生成一个字符串作为默认值
                    random_params[param_name] = "default_value"
                else:
                    # 根据参数类型生成随机值
                    random_params[param_name] = RandomInitForType.generate_random_value(param_type)
            
            seed = Seed()
            seed.set_from_params(random_params)
            self._inputs.append(seed)
        self._corpus_init_size = len(self._inputs)

        # self._seed_run_finished记录初始的种子是否被处理完成
        self._seed_run_finished = not self._inputs
        # 下一个使用的种子输入的下标
        self._seed_idx = 0
        self._save_corpus = dirs and os.path.isdir(dirs[0])
        # self._inputs.append([])
    def get_corpus_size(self):
        return self._corpus_init_size
    def get_inputs(self):
        return self._inputs

    def _add_seed(self, path, types):
        with open(path, 'r', encoding='utf-8') as f:
            # 读取所有行
            lines = f.readlines()
            # 去除每行的换行符，并存储到列表中
            data_list = [line.strip() for line in lines]
            npy_dict = {}
            no_npy_dict = {}
            image_var = []
            # 将输入文件中的npy全都读取出来放入npy_dict中
            for data in data_list:
                # print('data:',data)
                if '=' not in data:
                    continue
                param_name, value = data.strip().split('=', 1)
                param_type = types.get(param_name, None) if types else None
                if value.endswith('_x.npy'):
                    npy_dict[param_name] = np.load(value, allow_pickle=True)
                    image_var.append(param_name)
                elif value.endswith('_y.npy'):
                    npy_dict[param_name] = np.load(value, allow_pickle=True)
                elif value.endswith('.npy'):
                    npy_dict[param_name] = np.load(value, allow_pickle=True)
                elif value.endswith('_datax'):
                    npy_dict[param_name] = load_images_from_folder(value)
                    image_var.append(param_name)
                else:
                    no_npy_dict[param_name] = value

            
            if len(npy_dict) != 0:
                npy_array_size = next(iter(npy_dict.values())).shape[0]
                # 检查所有数组的大小是否一致
                all_same_size = all(array.shape[0] == npy_array_size for array in npy_dict.values())
                if not all_same_size:
                    raise ValueError("All arrays must have the same size")

                for i in range(npy_array_size):
                    cur_params = {}
                    for param_name in types.keys():
                        if param_name in npy_dict:
                            v = npy_dict[param_name][i]
                            if param_name in image_var:
                                cur_params[param_name] = Image.fromarray(np.array(v, dtype=np.uint8))
                            else:
                                cur_params[param_name] = v
                        else:
                            cur_params[param_name] = no_npy_dict[param_name]
                    # print("seed:", cur_params)
                    seed = Seed()
                    seed.set_from_file(cur_params, types)
                    
                    self._inputs.append(seed)
            else:
                seed = Seed()
                seed.set_from_file(no_npy_dict, types)
                self._inputs.append(seed)

    def _add_file(self, path):
        with open(path, 'rb') as f:
            bs = bytearray(f.read())
            string_data = bs.decode('utf-8')
            string_data = string_data.strip()
            lists = eval(string_data)
            self._inputs.append(lists)

    @property
    def length(self):
        return len(self._inputs)

    @staticmethod
    def _rand(n):
        if n < 2:
            return 0
        return _randbelow(n)

    # Exp2 generates n with probability 1/2^(n+1).
    @staticmethod
    def _rand_exp():
        rand_bin = bin(random.randint(0, 2**32-1))[2:]
        rand_bin = '0'*(32 - len(rand_bin)) + rand_bin
        count = 0
        for i in rand_bin:
            if i == '0':
                count +=1
            else:
                break
        return count

    @staticmethod
    def _choose_len(n):
        x = Corpus._rand(100)
        if x < 90:
            return Corpus._rand(min(8, n)) + 1     #返回1到8
        elif x < 99:
            return Corpus._rand(min(32, n)) + 1    #返回1到32
        else:
            return Corpus._rand(n) + 1        #返回1到n

    def put(self, buf):
        self._inputs.append(buf)


    def generate_input(self, batch_size, total_seedpool_size, target_func):
        '''
        首先选取一个从0到total_seedpool_size - batch_size的整数，作为batch_size个种子的起始坐标
        '''
        random_integer = random.randint(0, total_seedpool_size)
        if random_integer <= self._corpus_init_size - batch_size:
            bufs = self._inputs[random_integer:random_integer + batch_size]
            param_types = ParamsType.get_parameter_types(target_func)
            new_buf_params = {}
            for param_name, param_type in param_types.items():
                l = []
                for seed in bufs:
                    l.append(seed.get_params()[param_name])

                if param_type==list:
                    new_buf_params[param_name] = l
                else:
                    new_buf_params[param_name] = l[0]
            buf = Seed()
            buf.set_from_params(new_buf_params)
            buf.enum = bufs[0].enum
        else:
            with open(f'{config.LOCAL_SEED_POOL}/{random_integer}.pickle', 'rb') as f:
                buf = pickle.load(f)
        mutation = Mutation(self._max_input_size)
        buf = mutation.mutation(buf)
        buf.from_seed_que.append(random_integer)
        
        return buf, False
    

