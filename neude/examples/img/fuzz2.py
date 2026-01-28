import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'neude', 'neude'))
from neude.main import PythonFuzz
import numpy as np
@PythonFuzz
def fuzz(img):
    img = np.array(img)
    shape = img.shape
    return shape[0]

if __name__ == '__main__':
    fuzz()