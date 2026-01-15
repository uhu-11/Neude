import collections
import sys
import inspect
from pythonfuzz.Seed import Seed

prev_line = 0
prev_filename = ''
#data是存储路径的字典
data = collections.defaultdict(set)
func_lines_map = {}
all_func_lines = 0

'''
该函数进行调用路径的追踪存储
'''
def trace(frame, event, arg):
    if event != 'line':
        return trace

    global prev_line
    global prev_filename

    func_filename = frame.f_code.co_filename
    func_line_no = frame.f_lineno
    # 如果当前执行的语句和上一个执行的语句不是同一个python文件，则说明发生了文件的调用跳转，就以两个文件名为key，存储上一条语句的行号以及当前语句行号的元组
    if func_filename != prev_filename:
        # We need a way to keep track of inter-files transferts,
        # and since we don't really care about the details of the coverage,
        # concatenating the two filenames in enough.
        data[func_filename + prev_filename].add((prev_line, func_line_no))
    # 如果当前执行的语句和上一条执行的语句是同一个python文件，则以当前文件名为key，存储上一条语句的行号以及当前语句行号的元组
    else:
        data[func_filename].add((prev_line, func_line_no))

    prev_line = func_line_no
    prev_filename = func_filename

    # func_name = frame.f_code.co_name
    # module_name = frame.f_globals["__name__"]  # 获取模块名
    # key = module_name + " " + func_name
    # print(key)
    # if key not in func_lines_map.keys():
    #     try:
    #         # 计算并打印函数的有效代码行数
    #         lines_count = get_function_code_lines(func_name, sys.modules[module_name])
    #         func_lines_map[key] = lines_count
    #         all_func_lines += lines_count
    #     except ValueError as e:
    #         print(e)

    return trace


def get_coverage():
    # 返回执行的行数
    return sum(map(len, data.values()))

def get_lines():
    return data

def get_all_lines():
    return all_func_lines

def get_coverage_rate():
    get_coverage() / get_all_lines()

def get_function_code_lines(func_name, module):
    """根据函数名和模块，返回函数的有效代码行数（不包括注释和空行）。"""
    func = getattr(module, func_name, None)
    if func is None:
        raise ValueError(f"Function {func_name} not found in the module {module}")

    try:
        source_lines, starting_line_no = inspect.getsourcelines(func)
    except (OSError, TypeError):
        raise ValueError(f"Cannot retrieve source code for function {func_name}")

    non_empty_non_comment_lines = 0
    for line in source_lines:
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith("#"):
            non_empty_non_comment_lines += 1

    return non_empty_non_comment_lines

