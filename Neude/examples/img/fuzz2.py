import sys
sys.path.append('/media/lzq/D/lzq/pythonfuzz-pylot')
from pythonfuzz.main import PythonFuzz
import numpy as np
@PythonFuzz
def fuzz(img):
    img = np.array(img)
    shape = img.shape
    return shape[0]

if __name__ == '__main__':
    fuzz()