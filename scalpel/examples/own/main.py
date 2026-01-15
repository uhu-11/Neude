import ast
import random
import sys
sys.path.append('/media/lzq/D/lzq/pylot_test')

from scalpel.cfg import CFGBuilder

cfg = CFGBuilder().build_from_file('/media/lzq/D/lzq/pylot_test/pylot/pylot2.py', '/media/lzq/D/lzq/pylot_test/pylot/pylot2.py')

print("cfg_all_val", cfg.all_variable)


for block in cfg.get_all_blocks():
    # 判断该块的所有的statement中是否有这个function的调用
    for statement in block.statements:
        print(type(statement))
        # content = ast.dump(statement)
        # print(content)

res = cfg.build_visual('pdf')
# res.body[0] = '＜font color="blue"＞\n'+res.body[0]+'\n＜/font>'
print("RES_BODY", res.body[0])
print("RES_TYPE", type(res))
print("RES_ATTR", res.body[0])
if('model' in res.body[0]):
    print("YES")
# 生成整体的cfg


res.render('test-output/out', view=True)


for fun_name, row in cfg.function_firstRow.items():
    print(fun_name, " ", row)

for class_name, row in cfg.class_firstRow.items():
    print(class_name, " ", row)

print("----------------------------------------")
for fun_name, row_list in cfg.function_callRow.items():
    for row in row_list:
        print(fun_name, " ", row)
for class_name, row_list in cfg.class_callRow.items():
    for row in row_list:
        print(class_name, " ", row)







# for block in cfg.get_all_blocks():
#     # 判断该块的所有的statement中是否有这个function的调用
#     for statement in block.statements:
#         content = ast.dump(statement)
#         print(content)

# for fun_name, fun_cfg in cfg.class_cfgs.items():
#
#     print(fun_name)
#     print(fun_cfg)



# print(cfg.get_all_blocks())
# for block in cfg.get_all_blocks():
#     for link in block.exits:
#         print(link)
# 为fib函数生成cfg
# i = 0
print("XIANGGUAN:MAIN", cfg.all_variable)
for (block_id, fun_name), fun_cfg in cfg.functioncfgs.items():
    ... # do something
    # print(fun_cfg.__dict__)
    # for block in fun_cfg.get_all_blocks() :
    #     ind = block.at()
    #     print(ind)
    graph = fun_cfg.build_visual('png')
    graph.render("DNN "+fun_name, view=False)
    print("XIANGGUAN : ", fun_name, fun_cfg.all_variable)

for b in cfg.get_all_blocks():
    print(b.id, b.related_variables)


for statement in cfg.get_all_blocks()[0].statements:
    print(statement)

print(ast.dump(cfg.get_all_blocks()[0].statements[6].test))


# print(len(cfg.get_all_blocks()))

# for block in cfg.get_all_blocks():
#     calls = block.at()
#     print("-----------------------")
#     print(calls)

# def findVarInTuple(tuple):
#     res = []
#     def helper():
#         for ele in tuple.elts:
#             if isinstance(ele, ast.Name):
#                 res.append(ele.id)
#             if isinstance(ele, ast.Tuple):
#                 helper(ele)
#     return res
#
# for block in cfg.get_all_blocks():
#     #存储所有与load_model有关的变量名的list
#     all_variable= ["load_model"]
#     for statement in block.statements:
#         if isinstance(statement, ast.Assign):
#             content = ast.dump(statement)
#             for v in all_variable:
#                 if v in content:    #表示这个statement里面包含和load_model有关的变量
#
#
#
#
#             # 获取每个target中的target中的name变量
#             for target in statement.targets:
#                 if isinstance(target, ast.Tuple):
#                     all_targets += findVarInTuple(target)
#                 elif isinstance(target, ast.Name):
#                     all_targets.append(target.id)
#


