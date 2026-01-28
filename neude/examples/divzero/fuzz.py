
from neude.main import PythonFuzz


@PythonFuzz
def fuzz(num):
    num = int.from_bytes(num, 'big')
    if num >= 10:
        return num/10
    else:
        return 1/(num-1)

if __name__ == '__main__':
    fuzz()
