
from tensorflow.keras.datasets import mnist

(_, _), (x_test, y_test) = mnist.load_data()

print(x_test.shape)