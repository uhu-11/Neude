import random
class BoolMutationOperator:
    def __init__(self):
        self.ops = [
            self.get_random,
            self.reverse
        ]

    def getOps(self):
        return self.ops
    
    def get_random(self,value):
        return random.choice([True, False])
    
    def reverse(self,value):
        return not value
    