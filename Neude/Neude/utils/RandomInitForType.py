import random
import string

from pythonfuzz.utils import ParamsType


def generate_random_value(value_type):
    if value_type == int:
        return random.randint(0, 100)
    elif value_type == float:
        return random.uniform(0.0, 100.0)
    elif value_type == str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    elif value_type == bool:
        return random.choice([True, False])
    elif value_type == list:
        # For a 255x255x3 list representing an RGB image
        return [[[random.randint(0, 255) for _ in range(3)] for _ in range(255)] for _ in range(255)]
    elif value_type == dict:
        return {generate_random_value(str): generate_random_value(random.choice([int, float, str, bool])) for _ in range(5)}
    elif isinstance(value_type, list):
        return [generate_random_value(value_type[0]) for _ in range(5)]
    elif isinstance(value_type, tuple):
        return tuple(generate_random_value(t) for t in value_type)
    elif isinstance(value_type, type):
        return value_type()
    else:
        return None

