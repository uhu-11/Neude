import sys
sys.path.append('/media/lzq/D/pythonfuzz-pylot')
from pythonfuzz.main import PythonFuzz
import numpy as np
from PIL.Image import Image

@PythonFuzz
def fuzz(a: float):
    if a>0:
        a=1
    else:
        a=2
    return a

if __name__ == '__main__':
    fuzz()


