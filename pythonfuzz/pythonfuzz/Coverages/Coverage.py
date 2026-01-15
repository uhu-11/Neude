import inspect
import ast

class Coverage:

    def __init__(self, target):
        self.target = target
        self.total_lines = self.get_non_empty_non_comment_lines()
        self.fun_start_line=0
        self.status = []

    def get_coverage(selft):
        pass


    def get_coveraged_lines(self):
        pass



    #获取函数的所有行数
    def get_non_empty_non_comment_lines(self):
        source_lines, starting_line_no = inspect.getsourcelines(self.target)
        self.fun_start_line = starting_line_no
        source = ''.join(source_lines)
        tree = ast.parse(source)

        non_empty_non_comment_lines = 0
        in_docstring = False

        # 获取目标函数的名称
        target_function_name = self.target.__name__

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == target_function_name:
                func_start_lineno = node.lineno
                func_end_lineno = node.body[-1].lineno if node.body else func_start_lineno

                for lineno in range(func_start_lineno, func_end_lineno + 1):
                    line = source_lines[lineno - func_start_lineno]
                    stripped_line = line.strip()

                    if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                        in_docstring = not in_docstring

                    if not in_docstring and stripped_line and not stripped_line.startswith("#"):
                        non_empty_non_comment_lines += 1

        return non_empty_non_comment_lines
