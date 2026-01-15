import copy

from scalpel.SSA.const import SSA
from scalpel.cfg import CFGBuilder
from scalpel.core.mnode import MNode
from scalpel.cfg.model import Block, Link


def read_code(path):
    with open(file=path, mode='r', encoding='utf-8') as file:
        content = file.read()
    return content

#读取py文件中的代码
code_str = read_code('DNN.py')

#构建CFG
mnode = MNode("local") # create file name for given source code
mnode.source = code_str
mnode.gen_ast() # build abstract syntax tree
cfg = mnode.gen_cfg()
cfg = CFGBuilder().build_from_file('DNN.py', './DNN.py')
res = cfg.build_visual('pdf')

#构建SSA
m_ssa = SSA()
ssa_results, const_dict = m_ssa.compute_SSA(cfg)


def get_related_val(cfg):
    related_val = {}

    print(cfg.all_variable)

    for name, value in const_dict.items():
        if name[0] in cfg.all_variable:
            if name[0] not in related_val:
                related_val[name[0]] = {name[1]: value}
            else:
                v = related_val[name[0]]
                v[name[1]] = value
    return related_val

related_val = get_related_val(cfg)
for name, value in related_val.items():
    print(name, value)


# 寻找cfg的所有路径
def find_all_trace(cfg):
    all_traces = []
    start = cfg.entryblock
    def dfs(block, cur_trace):
        cur_trace.append(block)
        if len(block.exits) == 0:
            all_traces.append(list(cur_trace))
        for link in block.exits:
            cur_trace.append(link)
            dfs(link.target, cur_trace)
            cur_trace.pop()
        cur_trace.pop()
    dfs(start, cur_trace=[])
    return all_traces



# traces = find_all_trace(cfg)
# print("-------------------------")
# for i, trace in enumerate(traces):
#     print(i)
#     for n in trace:
#         print(n)

# 获得cfg所有与相关变量有关的路径
def find_related_trace(cfg):

    #声明一个字典存储与相关变量有关的路径，key为相关变量，value为一个list，其中存储着所有与key相关的路径
    related_trace = {}

    traces = find_all_trace(cfg)
    for trace in traces:
        trace_vals = set()
        for ele in trace:
            print("isinstance(trace, Block)", type(ele) == Block)
            if type(ele) == Block:
                print("相关变量", ele.related_variables)
                trace_vals = trace_vals | ele.related_variables    #获取这个路径中所有block中的相关变量
        for v in trace_vals:
            if v not in related_trace:
                related_trace[v] = [trace]
            else:
                related_trace[v].append(trace)
    return related_trace

related_trace = find_related_trace(cfg)
for val, traces in related_trace.items():
    print("相关变量", val)
    for i, trace in enumerate(traces):
        print("路径", i, trace)



'''
    将cfg中与related_var无关的边和节点都去除
    主要分为如下步骤：
        1. 先深copy一个新的cfg
        2. 得到与该related_var变量有关的所有的link和block的集合
        3. 遍历cfg，如果block和link不在第二步中所得的集合中的话，就将link从exits中删除
'''
def get_var_related_cfg(related_var, cfg):

    def remove_block(block, all_block_link):
        for ele in all_block_link:
            print(ele)
        for exit_link in block.exits:
            print(exit_link)
        for exit_link in block.exits:
            print("woshishabi", exit_link.target.id)

        ind = 0
        while ind < len(block.exits):
            exit_link = block.exits[ind]
            print("cur_exit_link", exit_link.target)
            if exit_link not in all_block_link:
                print(exit_link, "not in yesyes")
                block.exits.remove(exit_link)
            else:
                print(exit_link, "in yesyes")
                remove_block(exit_link.target, all_block_link)
                ind = ind+1



    #深拷贝一份cfg用于去除和related_var无关的边和节点
    var_related_cfg = copy.deepcopy(cfg)
    #获取cfg中与related_var有关的路径
    # print("哈哈哈哈哈哈",find_related_trace(cfg))
    traces = find_related_trace(var_related_cfg)[related_var]
    for t in traces:
        print("TtTt", t)

    all_block_link = set()
    #将traces中的所有的边和节点都存到all_block_link这个集合中
    for trace in traces:
        all_block_link = all_block_link | set(trace)

    for ele in all_block_link:
        if type(ele) == Block:
            print("BlockId", ele.id)
        else:
            print("linkTarget", ele.target.id)

    remove_block(var_related_cfg.entryblock, all_block_link)
    for block in var_related_cfg.finalblocks:
        if block not in all_block_link:
            var_related_cfg.finalblocks.remove(block)
    return var_related_cfg

related_cfg = get_var_related_cfg('all_predictions' ,cfg)
res = related_cfg.build_visual('pdf')
res.render('test-output/DNN', view=True)






