import sys
sys.path.append('/media/lzq/D/lzq/fuzz_tool')
sys.path.append('/media/lzq/D/lzq/fuzz_tool/pythonfuzz')
sys.path.append('/media/lzq/D/lzq/fuzz_tool/pythonfuzz/PTtool')
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from pythonfuzz.main import PythonFuzz
from Pillow.src.PIL import Image, ImageEnhance
import random


@PythonFuzz
def Image_Recognition(img, y:int, choice:int):

    img = img.resize((32, 32))  # 调整为 32x32 像素
    img = img.convert('RGB')  # 确保是 RGB 格式

    model = load_model('/media/lzq/D/lzq/fuzz_tool/pythonfuzz/examples/img/model_cifar_resNet20.h5')
   # 随机选择操作
    operations = ['rotate', 'flip', 'brightness', 'contrast', 'noise', 'crop', 'convert']
    operation = operations[choice % 7]
    print("operation", operation)


    if operation == 'rotate':
        angle = random.randint(0, 360)
        img = img.rotate(angle)

    elif operation == 'flip':
        if random.choice([True, False]):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
        else:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)  # 垂直翻转

    elif operation == 'brightness':
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.5, 1.5)  # 随机亮度因子
        img = enhancer.enhance(factor)

    elif operation == 'contrast':
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.5, 1.5)  # 随机对比度因子
        img = enhancer.enhance(factor)

    elif operation == 'noise':
        img_array = np.array(img) / 255.0
        noise = np.random.normal(0, 25, img_array.shape)  # 添加高斯噪声
        img_array = np.clip(img_array + noise, 0, 255)  # 确保值在0-255之间
        img = Image.fromarray(np.uint8(img_array))

    elif operation == 'crop':
        width, height = img.size
        left = random.randint(0, width // 4)
        top = random.randint(0, height // 4)
        right = random.randint(3 * width // 4, width)
        bottom = random.randint(3 * height // 4, height)
        if right > left and bottom > top:
            img = img.crop((left, top, right, bottom))  # 执行裁剪操作
        else:
            print("裁剪区域无效，未执行裁剪。")

    img = img.resize((32, 32))  # 调整为 32x32 像素
    img = img.convert('RGB')  # 确保是 RGB 格式
    img_array = np.array(img) / 255.0  # 归一化到 [0, 1] 范围
    img_array = np.expand_dims(img_array, axis=0)  # 变为 (1, 32, 32, 3)
    y_predict = model.predict(img_array)
    label = np.argmax(y_predict)
    confidence = np.max(y_predict)
    print("confidence:", confidence)
    score = 1 / round(confidence + 0.2)
    print("y:", y)
    if label == 0:
        print("it is a airplane")
        #confidence
    elif label == 1:
        print("it is a automobile")
    elif label == 2:
        print("it is a bird")
    elif label == 3:
        print("it is a cat")
    elif label == 4:
        print("it is a deer")
    elif label == 5:
        print("it is a dog")
    elif label == 6:
        print("it is a frog")
    elif label == 7:
        print("it is a horse")
    elif label == 8:
        print("it is a ship")
    else:
        print("it is a truck")
    del model

    
    return y_predict

if __name__ == '__main__':
    Image_Recognition()