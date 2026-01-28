import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'neude', 'neude'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'neude', 'PTtool'))
from PIL import Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from neude.main import PythonFuzz
from neude.Coverages.PTCoverage import PTCoverage
import inspect
pt = PTCoverage()
from PIL import Image

def test1(img:str, y:int):
    y = np.array([y])
    img = Image.open(img)
    img = img.resize((32, 32))  # 调整为 32x32 像素
    img = img.convert('RGB')  # 确保是 RGB 格式
    img_array = np.array(img) / 255.0  # 归一化到 [0, 1] 范围

    # 添加批次维度
    img = np.expand_dims(img_array, axis=0)  # 变为 (1, 32, 32, 3)
    
    model = load_model('/media/lzq/D/lzq/fuzz_tool/pythonfuzz/examples/img/model_cifar_resNet20.h5')

    y_predict = model.predict(img)

    del model
    return y_predict

def npy_test(a, b):
    print(a, b)
if __name__ == '__main__':
    img = Image.open('/media/lzq/D/lzq/fuzz_tool/pythonfuzz/examples/img/bird.png')
    img = np.array(img)
    
    print(img)

