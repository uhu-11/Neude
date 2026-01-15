import os
import math
import random
import struct
import hashlib


INTERESTING8 = [-128, -1, 0, 1, 16, 32, 64, 100, 127]
INTERESTING16 = [0, 128, 255, 256, 512, 1000, 1024, 4096, 32767, 65535]
INTERESTING32 = [0, 1, 32768, 65535, 65536, 100663045, 2147483647, 4294967295]

try:
    from random import _randbelow
except ImportError:
    from random import _inst
    _randbelow = _inst._randbelow

class StringMutationOperator:

    def __init__(self, max_input_size):
        self._max_input_size = max_input_size

    @staticmethod
    def _rand(n):
        if n < 2:
            return 0
        return _randbelow(n)

    @staticmethod
    def _choose_len(n):
        x = StringMutationOperator._rand(100)
        if x < 90:
            return StringMutationOperator._rand(min(8, n)) + 1
        elif x < 99:
            return StringMutationOperator._rand(min(32, n)) + 1
        else:
            return StringMutationOperator._rand(n) + 1

    @staticmethod
    def copy(src, dst, start_source, start_dst, end_source=None, end_dst=None):
        end_source = len(src) if end_source is None else end_source
        end_dst = len(dst) if end_dst is None else end_dst
        byte_to_copy = min(end_source - start_source, end_dst - start_dst)
        dst[start_dst:start_dst + byte_to_copy] = src[start_source:start_source + byte_to_copy]

    ##ops函数接收一个字符串
    def ops1(self, buf):
        res = bytearray(buf, 'utf-8')
        pos0 = self._rand(len(res))
        pos1 = pos0 + self._choose_len(len(res) - pos0)
        self.copy(res, res, pos1, pos0)
        res = res[:len(res) - (pos1 - pos0)]
        res = res.decode('utf-8')

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops2(self, buf):
        res = bytearray(buf, 'utf-8')
        pos = self._rand(len(res) + 1)
        n = self._choose_len(10)
        for k in range(n):
            res.append(0)
        self.copy(res, res, pos, pos + n)
        for k in range(n):
            res[pos + k] = self._rand(256)
        res = res.decode('utf-8')

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops3(self, buf):
        res = bytearray(buf, 'utf-8')
        if len(res) <= 1:
            return None
        src = self._rand(len(res))
        dst = self._rand(len(res))
        while src == dst:
            dst = self._rand(len(res))
        n = self._choose_len(len(res) - src)
        tmp = bytearray(n)
        self.copy(res, tmp, src, 0)
        for k in range(n):
            res.append(0)
        self.copy(res, res, dst, dst + n)
        for k in range(n):
            res[dst + k] = tmp[k]

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops4(self, buf):
        res = bytearray(buf, 'utf-8')
        if len(res) <= 1:
            return None
        src = self._rand(len(res))
        dst = self._rand(len(res))
        while src == dst:
            dst = self._rand(len(res))
        n = self._choose_len(len(res) - src)
        self.copy(res, res, src, dst, src + n)

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops5(self, buf):
        res = bytearray(buf, 'utf-8')
        if len(res) == 0:
            return None
        pos = self._rand(len(res))
        res[pos] ^= 1 << self._rand(8)

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops6(self, buf):
        res = bytearray(buf, 'utf-8')
        if len(res) == 0:
            return None
        pos = self._rand(len(res))
        res[pos] ^= self._rand(255) + 1

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops7(self, buf):
        res = bytearray(buf, 'utf-8')
        if len(res) <= 1:
            return None
        src = self._rand(len(res))
        dst = self._rand(len(res))
        while src == dst:
            dst = self._rand(len(res))
        res[src], res[dst] = res[dst], res[src]

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops8(self, buf):
        res = bytearray(buf, 'utf-8')
        if len(res) == 0:
            return None
        pos = self._rand(len(res))
        v = self._rand(2 ** 8)
        res[pos] = (res[pos] + v) % 256

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops9(self, buf):
        res = bytearray(buf, 'utf-8')
        if len(res) < 2:
            return None
        pos = self._rand(len(res) - 1)
        v = self._rand(2 ** 16)
        if bool(random.getrandbits(1)):
            v = struct.pack('>H', v)
        else:
            v = struct.pack('<H', v)
        res[pos] = (res[pos] + v[0]) % 256
        res[pos + 1] = (res[pos] + v[1]) % 256

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops10(self, buf):
        res = bytearray(buf, 'utf-8')
        if len(res) < 4:
            return None
        pos = self._rand(len(res) - 3)
        v = self._rand(2 ** 32)
        if bool(random.getrandbits(1)):
            v = struct.pack('>I', v)
        else:
            v = struct.pack('<I', v)
        res[pos] = (res[pos] + v[0]) % 256
        res[pos + 1] = (res[pos + 1] + v[1]) % 256
        res[pos + 2] = (res[pos + 2] + v[2]) % 256
        res[pos + 3] = (res[pos + 3] + v[3]) % 256

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops11(self, buf):
        res = bytearray(buf, 'utf-8')
        if len(res) < 8:
            return None
        pos = self._rand(len(res) - 7)
        v = self._rand(2 ** 64)
        if bool(random.getrandbits(1)):
            v = struct.pack('>Q', v)
        else:
            v = struct.pack('<Q', v)
        res[pos] = (res[pos] + v[0]) % 256
        res[pos + 1] = (res[pos + 1] + v[1]) % 256
        res[pos + 2] = (res[pos + 2] + v[2]) % 256
        res[pos + 3] = (res[pos + 3] + v[3]) % 256
        res[pos + 4] = (res[pos + 4] + v[4]) % 256
        res[pos + 5] = (res[pos + 5] + v[5]) % 256
        res[pos + 6] = (res[pos + 6] + v[6]) % 256
        res[pos + 7] = (res[pos + 7] + v[7]) % 256

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops12(self, buf):
        res = bytearray(buf, 'utf-8')
        if len(res) == 0:
            return None
        pos = self._rand(len(res))
        res[pos] = INTERESTING8[self._rand(len(INTERESTING8))] % 256

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops13(self, buf):
        res = bytearray(buf, 'utf-8')
        if len(res) < 2:
            return None
        pos = self._rand(len(res) - 1)
        v = random.choice(INTERESTING16)
        if bool(random.getrandbits(1)):
            v = struct.pack('>H', v)
        else:
            v = struct.pack('<H', v)
        res[pos] = v[0] % 256
        res[pos + 1] = v[1] % 256

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops14(self, buf):
        res = bytearray(buf, 'utf-8')
        if len(res) < 4:
            return None
        pos = self._rand(len(res) - 3)
        v = random.choice(INTERESTING32)
        if bool(random.getrandbits(1)):
            v = struct.pack('>I', v)
        else:
            v = struct.pack('<I', v)
        res[pos] = v[0] % 256
        res[pos + 1] = v[1] % 256
        res[pos + 2] = v[2] % 256
        res[pos + 3] = v[3] % 256

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def ops15(self, buf):
        res = bytearray(buf, 'utf-8')
        digits = []
        for k in range(len(res)):
            if ord('0') <= res[k] <= ord('9'):
                digits.append(k)
        if len(digits) == 0:
            return None
        pos = self._rand(len(digits))
        was = res[digits[pos]]
        now = was
        while was == now:
            now = self._rand(10) + ord('0')
        res[digits[pos]] = now

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]

        return res

    def getOps(self):
        return [getattr(self, 'ops1'), getattr(self, 'ops2'), getattr(self, 'ops3'), getattr(self, 'ops4'),
                getattr(self, 'ops5'), getattr(self, 'ops6'), getattr(self, 'ops7'), getattr(self, 'ops8'),
                getattr(self, 'ops9'), getattr(self, 'ops10'), getattr(self, 'ops11'), getattr(self, 'ops12'),
                getattr(self, 'ops13'), getattr(self, 'ops14'), getattr(self, 'ops15')]