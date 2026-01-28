# import unittest
# import zipfile
# import io
# from unittest.mock import patch

# import pythonfuzz

# class TestFindCrash(unittest.TestCase):
#     def test_find_crash(self):
#         def fuzz(buf):
#             f = io.BytesIO(buf)
#             z = zipfile.ZipFile(f)
#             z.testzip()

#         with patch('logging.Logger.info') as mock:
#             pythonfuzz.fuzzer.Fuzzer(fuzz).start()
#             self.assertTrue(mock.called_once)

from tensorflow.keras.datasets import cifar10
import numpy as np
# from tensorflow.keras.models import load_model
# model = load_model('/media/lzq/D/lzq/fuzz_tool/pythonfuzz/examples/img/model_cifar_resNet20.h5')

# x_train = np.load("/home/lzq/x_train.npy")
# y_train = np.load("/home/lzq/y_train.npy")
# x_train = x_train.reshape(-1,32,32,3).astype('float32') / 255.0
# y_train = y_train.reshape(-1)

# y_predict = np.argmax(model.predict(x_train), axis=1)
# print(y_predict)
# print(y_train)
# # np.save("/home/lzq/x_train.npy", x_train)
# # np.save("/home/lzq/y_train.npy", y_train)


import numpy as np

# 指定 .npy 文件的路径
npy_file_path = '/home/lzq/obstacles_dataset/obstacles_y.npy'

# 读取 .npy 文件
data = np.load(npy_file_path,allow_pickle=True)
print(type(data[0][0]))