import sys
import os
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'PTtool'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

def fuzz(img:str, model_path:str, y:int):
    print("img:", img)
    print("model_path:", model_path)
    print("y:", y)
    img = Image.open(img)
    img = img.resize((32, 32))  # 调整为 32x32 像素
    img = img.convert('RGB')  # 确保是 RGB 格式
    img_array = np.array(img) / 255.0  # 归一化到 [0, 1] 范围

    # 添加批次维度
    img = np.expand_dims(img_array, axis=0)  # 变为 (1, 32, 32, 3)

    model = load_model(model_path)
    print("img.shape:", img.shape)

    y_predict = model.predict(img)
    return y_predict

if __name__ == '__main__':
    fuzz("/media/lzq/D/lzq/pythonfuzz-pylot/examples/img/airplane.png", "/media/lzq/D/lzq/pythonfuzz-pylot/examples/img/model_cifar_resNet20.h5", 0)

