import ast
import inspect
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'neude', 'neude'))

class PredictCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.predict_calls = []

    def visit_Call(self, node):
        # 检查调用的函数名是否为 'predict'
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'predict':
                # 获取调用者对象，例如 model.predict
                value = node.func.value
                if isinstance(value, ast.Name):
                    caller = value.id
                elif isinstance(value, ast.Attribute):
                    # 处理如 self.model.predict 的情况
                    caller = value.attr
                else:
                    caller = None
                self.predict_calls.append((caller, node.lineno))
        self.generic_visit(node)

def function_uses_predict(func):
    """
    检查给定函数内部是否调用了 'predict' 方法。

    :param func: 需要检查的函数对象
    :return: 布尔值，表示是否调用了 'predict'
    """
    try:
        source = inspect.getsource(func)
    except OSError:
        print("无法获取函数源代码。")
        return False

    tree = ast.parse(source)
    visitor = PredictCallVisitor()
    visitor.visit(tree)

    if visitor.predict_calls:
        '''print(f"函数 '{func.__name__}' 内调用了 'predict' 方法。调用详情：")
        for caller, lineno in visitor.predict_calls:
            caller_str = f"{caller}." if caller else ""
            print(f"  - {caller_str}predict() 在第 {lineno} 行")'''
        return True
    else:
        '''print(f"函数 '{func.__name__}' 内未调用 'predict' 方法。")'''
        return False

