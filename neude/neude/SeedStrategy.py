

try:
    from random import _randbelow
except ImportError:
    from random import _inst
    _randbelow = _inst._randbelow


class SeedStrategy:

    @staticmethod
    def _rand(n):
        if n < 2:
            return 0
        return _randbelow(n)

    @staticmethod
    def random_select(inputs):
        return inputs[SeedStrategy._rand(len(inputs))]



