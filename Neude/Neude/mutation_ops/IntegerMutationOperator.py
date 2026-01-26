import random
class IntegerMutationOperator:
    def __init__(self):
        self.ops = [
            # self.negate,           # 取反
            self.add_random,       # 加一个随机数
            # self.subtract_random,  # 减一个随机数
            # self.multiply_small,   # 乘以一个小数
            self.interesting_value, # 使用一些特殊的边界值
            self.get_random
        ]

    def getOps(self):
        return self.ops

    # def negate(self, value):
    #     return -value

    def add_random(self, value):
        return value + random.randint(0, 100)
    def get_random(self, value):
        return random.randint(0,100)

    # def subtract_random(self, value):
    #     return value - random.randint(-100, 100)

    # def multiply_small(self, value):
    #     multipliers = [2, 4, 8, 16, 32, 64]
    #     return value * random.choice(multipliers)

    def interesting_value(self, value):
        interesting_values = [
            0,           # 零
            1,           # 一
            # -1,          # 负一
            # 2**31-1,    # INT_MAX
            # # -2**31,     # INT_MIN
            # 2**16,      # 较大的2的幂
            # 2**8,       # 256
            127,        # 一字节最大值
            128,        # 一字节最大值+1
            255,        # 一字节无符号最大值
            256,        # 一字节无符号最大值+1
        ]
        return random.choice(interesting_values)