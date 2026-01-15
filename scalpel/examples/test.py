import random

import numpy as np
from random import sample
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# 加载预训练模型
loaded_model = load_model('lenet5_mnist_model.h5')

# 加载Mnist数据集
(_, _), (x_test, y_test) = mnist.load_data()

# 将数据归一化并转换为float类型
x_test = x_test.astype('float32') / 255

# 添加一个维度来匹配LeNet5模型的输入维度
x_test = np.expand_dims(x_test, axis=-1)


# 德州扑克规则判断函数
def texas_holdem(hand):
    #计算手牌的值和置信度
    values = []
    confidences = []
    for img in hand:
        prediction = loaded_model.predict(np.expand_dims(img, axis=0))

        values.append(np.argmax(prediction))
        confidences.append(np.max(prediction))

    value = sum(values)

    if value > 18:
        return 1    # 一对
    else:
        return 0  # 散牌


# 生成随机手牌函数
def generate_hand(dataset, indices):
    return [dataset[i] for i in indices]


# 初始资金
initial_funds = 10000

# 生成两组随机手牌
indices_hand1 = sample(range(len(x_test)), 5)
indices_hand2 = sample(range(len(x_test)), 5)

hand1 = generate_hand(x_test, indices_hand1)
hand2 = generate_hand(x_test, indices_hand2)

# 输出两组手牌的数字标签
print("手牌1:", [np.argmax(loaded_model.predict(np.expand_dims(img, axis=0))) for img in hand1])
print("手牌2:", [np.argmax(loaded_model.predict(np.expand_dims(img, axis=0))) for img in hand2])


# 比较两组手牌大小
def compare_hands(hand1, hand2, funds):
    rank_hand1, value_hand1, confidence_hand1 = texas_holdem(hand1)
    rank_hand2, value_hand2, confidence_hand2 = texas_holdem(hand2)


    if rank_hand1 > rank_hand2:
        return "手牌1获胜！奖励: {}".format(reward), funds
    else:
        return "双方平局！", funds


compare_hands(hand1, hand2, initial_funds)
result, funds = compare_hands(hand1, hand2, initial_funds)
print(result)
print("当前资金:", funds)

def a(i:int):
    if 2>2:
	    return 1
    else:
	    return 0