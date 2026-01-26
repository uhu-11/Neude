import math
import random
import struct

class NumberMutationOperator:
    '''
    对当前值进行乘法变异
    '''
    def ops1(self, value):
        mutation_factor = random.uniform(0.5, 1.5)  # 生成一个在0.5到1.5之间的随机变异因子
        mutated_decimal = value * mutation_factor  # 以随机变异因子进行乘法变异
        mutated_decimal = min(1.0, max(0.0, mutated_decimal))  # 将结果限制在0到1之间
        return mutated_decimal

    '''
    对当前值应用一个随机指数变换，使其更接近 0 或 1。
    '''
    def ops2(self, value):
        factor = random.uniform(0.8, 1.2)
        new_value = value ** factor
        return max(0.0, min(1.0, new_value))

    '''
    对当前值采用加法的方式添加一个随机的微小扰动
    '''
    def ops3(self, value):
        perturbation = random.uniform(-0.1, 0.1)
        new_value = value + perturbation
        return max(0.0, min(1.0, new_value))
    '''
    使用高斯分布生成一个浮点数（平均值为当前值，标准差为0.1）
    '''
    def ops4(self, value):
        new_value = random.gauss(value, 0.1)
        return max(0.0, min(1.0, new_value))
    '''
    对当前值进行反转
    '''
    def ops5(self, value):
        new_value = 1.0 - value
        return new_value

    '''
    在浮点数的位级上进行变异，可以改变它的部分位，以探索不同的浮点数表示
    '''

    def ops6(self, value):
        # Convert float to bytes
        bytes_value = struct.pack('d', value)
        # Convert bytes to integer
        int_value = int.from_bytes(bytes_value, byteorder='little', signed=False)

        # Perform bitwise mutation
        mutation_point = random.randint(0, 63)  # Assuming 64-bit double
        mutated_value = int_value ^ (1 << mutation_point)

        # Convert back to float
        mutated_bytes = mutated_value.to_bytes(8, byteorder='little', signed=False)
        new_value = struct.unpack('d', mutated_bytes)[0]

        # Ensure the mutated value is within [0, 1]
        new_value = max(0.0, min(1.0, new_value))

        return new_value
    '''
    进行三角函数变化
    '''
    def ops7(self, value):
        new_value = math.sin(math.pi * value)
        return max(0.0, min(1.0, new_value))
    '''
    开平方
    '''
    def ops8(self, value):
        new_value = math.sqrt(value)
        return max(0.0, min(1.0, new_value))
    '''
    使用sigmoid函数
    '''
    def ops9(self, value):
        new_value = 1 / (1 + math.exp(-value))
        return new_value
    def getOps(self):
        return [getattr(self, 'ops1'), getattr(self, 'ops2'), getattr(self, 'ops3'), getattr(self, 'ops4'),
                getattr(self, 'ops5'), getattr(self, 'ops6'), getattr(self, 'ops7'), getattr(self, 'ops8'),
                getattr(self, 'ops9')]



