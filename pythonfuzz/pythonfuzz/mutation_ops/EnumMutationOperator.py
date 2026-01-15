from random import choice

class EnumMutationOperator:

    def ops(self, eles):
        eles = list(eles)
        return choice(eles)

    def getOps(self):
        return [getattr(self, 'ops')]