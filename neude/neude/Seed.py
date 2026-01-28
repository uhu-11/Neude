import inspect
import os.path
from PIL import Image
import numpy as np
import random
import time
from neude.config import GENERATED_IMAGES_PATH
import ast
class Seed:
    def __init__(self):
        self.params = {}  
        self.enum = {}
        #标识这个种子是哪个种子变以来的
        self.from_seed_que = []

    def set_from_file(self, params_dict, param_types):
        # for k, value in params_dict.items():
        #     print(f"key:{k}, Value: {value}, Type: {type(value)}")
        for param_name, param_type in param_types.items():
            value = params_dict[param_name]
            if value is not None:
                if param_name.endswith("_enum"):
                    self.enum[param_name] = ast.literal_eval(value)
                    self.params[param_name] = random.choice(self.enum[param_name])
                # 根据类型转换值
                elif param_name == 'y' or param_name.endswith('_label'):
                    self.params[param_name] = value
                elif param_type == Image.Image:
                    if type(value) == np.ndarray:
                        self.params[param_name] = Image.fromarray(value)
                    else:
                        self.params[param_name] = Image.open(value)
                elif type(value) == Image.Image:
                    if param_type == str:
                        timestamp = str(int(time.time())) # 获取当前时间
                        if not os.path.exists(GENERATED_IMAGES_PATH):
                            os.makedirs(GENERATED_IMAGES_PATH)
                        output_path = os.path.join(GENERATED_IMAGES_PATH, f"{timestamp}.png")  # 生成保存路径
                        value.save(output_path)
                        self.params[param_name] = output_path
                    else:
                        self.params[param_name] = value
                elif param_type is None and os.path.exists(value) and value.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    img = Image.open(value)
                    self.params[param_name] = img
                
                elif param_type == np.ndarray and type(value)!=np.ndarray:
                    self.params[param_name] = np.array(value)
                elif type(value)==np.ndarray:
                    self.params[param_name] = value
                elif param_type == bool:
                    if value == 'True':
                        self.params[param_name] = True
                    else:
                        self.params[param_name] = False
                elif param_type is not None and param_type != inspect._empty:
                    self.params[param_name] = param_type(value)
                else:
                    # 自动类型推断
                    if isinstance(value, int):
                        self.params[param_name] = int(value)
                    elif self.is_float(value):
                        self.params[param_name] = float(value)
                    elif value == "True":
                        self.params[param_name] = True
                    elif value == "False":
                        self.params[param_name] = False
                    else:
                        self.params[param_name] = value
            else:
                if param_type is None:
                    self.params[param_name] = None
                else:
                    self.params[param_name] = param_type()


  

    def set_from_params(self, params_dict):
        self.params = params_dict

    def get_params(self):
        return self.params
    def get_enum(self):
        return self.enum

    def is_float(self, s):
        try:
            float(s)
            return True
        except Exception as e:
            return False
