# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import random

# 加载Mnist数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 将数据归一化并转换为float类型
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 将标签进行独热编码
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 添加一个维度来匹配LeNet5模型的输入维度
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 定义LeNet5模型
model = models.Sequential([
    layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# 保存模型
model.save('lenet5_mnist_model.h5')

from tensorflow.keras.models import load_model

# 加载模型
# loaded_model = load_model('lenet5_mnist_model.h5')

# 使用加载的模型进行预测
# all_predictions = loaded_model.predict(x_test)
# all_predictions = 1
# if all_predictions > 1:
#     all_predictions = 6
# else:
#     a = 10
# # predictions = all_predictions[:100]


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
def texas_holdem(hand: list):
    # 计算手牌的值和置信度
    values = []
    confidences = []
    for img in hand:
        prediction = loaded_model.predict(np.expand_dims(img, axis=0))

        values.append(np.argmax(prediction))
        confidences.append(np.max(prediction))

    value = sum(values)

    # 判断是否为顺子
    def is_straight(values):
        return max(values) - min(values) == 4 and len(set(values)) == 5

    # 判断是否为同花
    def is_flush(suits):
        return len(set(suits)) == 1

    suits = [i // (len(hand) // 4) for i in range(len(hand))]  # 根据位置下标定义花色

    if is_straight(values) and is_flush(suits):
        return 8, value, max(confidences)  # 同花顺
    elif values.count(max(values, key=values.count)) == 4:
        return 7, value, max(confidences)  # 四条
    elif sorted(values)[-2:] == sorted(values)[:2] or sorted(values)[-3:] == sorted(values)[:3]:
        return 6, value, max(confidences)  # 葫芦
    elif is_flush(suits):
        return 5, value, max(confidences)  # 同花
    elif is_straight(values):
        return 4, value, max(confidences)  # 顺子
    elif values.count(max(values, key=values.count)) == 3:
        return 3, value, max(confidences)  # 三条
    elif values.count(max(values, key=values.count)) == 2 and len(set(values)) == 3:
        return 2, value, max(confidences)  # 两对
    elif values.count(max(values, key=values.count)) == 2:
        return 1, value, max(confidences)  # 一对
    else:
        return 0, value, max(confidences)  # 散牌


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

    # 计算置信度差值
    confidence_diff = abs(confidence_hand1 - confidence_hand2)

    # 计算奖励
    if confidence_diff >= 0.2:
        reward = 1 / (confidence_diff - 0.2)  # 根据公式计算奖励
    elif confidence_diff >= 0.4:
        reward = 4 / (confidence_diff - 0.4)  # 根据公式计算奖励
    elif confidence_diff <= 0.0000001:
        reward = 1 / (confidence_diff - 0.0000001)  # 根据公式计算奖励
    else:
        reward = 1  # 置信度差值小于0.2时固定奖励

    if rank_hand1 > rank_hand2:
        funds += reward
        return "手牌1获胜！奖励: {}".format(reward), funds
    elif rank_hand1 < rank_hand2:
        funds -= reward
        return "手牌2获胜！损失: {}".format(reward), funds
    elif value_hand1 > value_hand2:
        funds += reward
        return "手牌1获胜！奖励: {}".format(reward), funds
    elif value_hand1 < value_hand2:
        funds -= reward
        return "手牌2获胜！损失: {}".format(reward), funds
    else:
        return "双方平局！", funds


compare_hands(hand1, hand2, initial_funds)
result, funds = compare_hands(hand1, hand2, initial_funds)
print(result)
print("当前资金:", funds)


# 输入一个图片，输出一个图片，看一下是否能生成图片测试用例，
# 有的话写一个自己的测试生成逻辑
# 一个函数，光加载模型，然后光打印模型信息
# 输入一个图片，读入模型，模型预测图片
# 测试其他的逻辑部分
# 能不能限制生成的参数
