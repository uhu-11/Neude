import os
import random
import time
import numpy as np
from PIL import Image

from neude.Seed import Seed
from neude.mutation_ops.ImageMutationOperator import ImageMutationOperator
from neude.mutation_ops.NumberMutationOperator import NumberMutationOperator
from neude.mutation_ops.StringMutationOperator import StringMutationOperator
from neude.mutation_ops.IntegerMutationOperator import IntegerMutationOperator
from neude.mutation_ops.BoolMutationOperator import BoolMutationOperator
from neude.config import GENERATED_IMAGES_PATH,USE_FUNCTIONS

class Mutation:
    def __init__(self, max_input_size):
        self.numberMutationOperator = NumberMutationOperator()
        self.stringMutationOperator = StringMutationOperator(max_input_size)
        self.imageMutationOperator = ImageMutationOperator()
        self.integerMutationOperator = IntegerMutationOperator()
        self.boolMutationOperator = BoolMutationOperator()
        
        self.numberOperators = self.numberMutationOperator.getOps()
        self.stringOperators = self.stringMutationOperator.getOps()
        self.imageOperators = self.imageMutationOperator.getOps()
        if USE_FUNCTIONS == 'pythonfuzz':
            self.imageOperators = self.imageMutationOperator.getOps_pythonfuzz()
        elif USE_FUNCTIONS == 'deephunter':
            self.imageOperators = self.imageMutationOperator.getOps_deephunter()
        else:
            self.imageOperators = self.imageMutationOperator.getOps()
        self.integerOperators = self.integerMutationOperator.getOps()
        self.boolOperators = self.boolMutationOperator.getOps()

    def mutation(self, seed):
        mutated_params = {}
        for param_name, value in seed.get_params().items():
            if param_name in seed.get_enum():
                mutated_params[param_name] = random.choice(seed.get_enum()[param_name])
            elif value is True or value is False:
                ind = random.randint(0, len(self.boolOperators)-1)
                ops = self.boolOperators[ind]
                mutated_params[param_name] = ops(value)
            elif param_name == "y" or param_name.endswith('_label'):
                mutated_params[param_name] = value
            elif isinstance(value, int):     
                ind = random.randint(0, len(self.integerOperators)-1)
                ops = self.integerOperators[ind]
                mutated_params[param_name] = ops(value)
            elif isinstance(value, float):
                ind = random.randint(0, len(self.numberOperators)-1)
                ops = self.numberOperators[ind]
                mutated_params[param_name] = ops(value)
            elif isinstance(value, list) and isinstance(value[0], Image.Image):
                new_imgs=[]
                for img in value:
                    image_array = np.array(img)
                    img = image_array.tolist()
                    ind = random.randint(0, len(self.imageOperators) - 1)
                    ops = self.imageOperators[ind]
                    image = Image.fromarray(np.array(ops(img), dtype=np.uint8))
                    new_imgs.append(image)
                mutated_params[param_name] = new_imgs
            elif isinstance(value, Image.Image):
                image_array = np.array(value)
                img = image_array.tolist()
                ind = random.randint(0, len(self.imageOperators) - 1)
                ops = self.imageOperators[ind]
                image = Image.fromarray(np.array(ops(img), dtype=np.uint8))
                mutated_params[param_name] = image
            elif isinstance(value, np.ndarray):
                img = value.tolist()
                ind = random.randint(0, len(self.imageOperators) - 1)
                ops = self.imageOperators[ind]
                image = Image.fromarray(np.array(ops(img), dtype=np.uint8))
                mutated_params[param_name] = image
            elif isinstance(value, list):
                ind = random.randint(0, len(self.imageOperators)-1)
                ops = self.imageOperators[ind]
                mutated_params[param_name] = ops(value)
            elif isinstance(value, str) and os.path.exists(value) and value.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image = Image.open(value)
                image_array = np.array(image)
                img = image_array.tolist()
                ind = random.randint(0, len(self.imageOperators) - 1)
                ops = self.imageOperators[ind]
                image = Image.fromarray(np.array(ops(img), dtype=np.uint8))
                
                timestamp = str(int(time.time())) # 获取当前时间
                file_extension = os.path.splitext(value)[1]  # 获取原始文件后缀
                if not os.path.exists(GENERATED_IMAGES_PATH):
                    os.makedirs(GENERATED_IMAGES_PATH)
                output_path = os.path.join(GENERATED_IMAGES_PATH, f"{timestamp}{file_extension}")  # 生成保存路径
                image.save(output_path)
                mutated_params[param_name] = output_path
            else:
                ind = random.randint(0, len(self.stringOperators)-1)
                ops = self.stringOperators[ind]
                mutated_params[param_name] = ops(str(value))
        res = Seed()
        res.set_from_params(mutated_params)
        res.enum = seed.get_enum()
        return res
