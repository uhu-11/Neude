import sys
sys.path.append('/root/my_pythonfuzz/pythonfuzz_master')
from pythonfuzz.main import PythonFuzz
from tensorflow.keras.models import load_model
import numpy as np


@PythonFuzz
def fuzz(img):
    # print("img.type:=", type(img))
    loaded_model = load_model('lenet5_mnist_model.h5')
    img = np.array(img)
    x_test = img.astype('float32') / 255
    x_test = np.expand_dims(x_test, axis=-1)
    all_predictions = loaded_model.predict(x_test)
    predictions = all_predictions[:100]
    # print('predictions.type is', type(predictions[0]))
    num = np.argmax(predictions[0])
    print(num)
    raise Exception(f'test_error')

if __name__ == '__main__':
    fuzz()