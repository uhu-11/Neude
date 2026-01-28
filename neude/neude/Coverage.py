from neude import tracer
import inspect
import ast

class Coverage:

    #获取行覆盖
    @staticmethod
    def get_line_coverage(target):
        return tracer.get_coverage() / Coverage.get_non_empty_non_comment_lines(target)

    @staticmethod
    def get_line_coverage():
        return tracer.get_coverage()

    @staticmethod
    def get_coveraged_lines():
        return tracer.get_lines()




    #获取函数的所有行数
    @staticmethod
    def get_non_empty_non_comment_lines(func):
        source_lines, _ = inspect.getsourcelines(func)
        source = ''.join(source_lines)
        tree = ast.parse(source)

        non_empty_non_comment_lines = 0
        in_docstring = False

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_source_lines = inspect.getsourcelines(node)[0]
                for line in func_source_lines:
                    stripped_line = line.strip()

                    if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                        in_docstring = not in_docstring

                    if not in_docstring and stripped_line and not stripped_line.startswith("#"):
                        non_empty_non_comment_lines += 1

        return non_empty_non_comment_lines