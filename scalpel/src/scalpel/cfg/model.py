"""
The module provides interface for CFG Class, which is a control flow graph (CFG) representation of a Python function or module.
It is a directed graph with basic blocks as nodes and control flow jumps as edges. The CFG class can be used to analyze the control flow of a Python function or module. 
It can also be used to build a "graphviz" visualization of the CFG, which can be helpful for understanding the control flow of a complex function or module.
"""

import ast
import re
import sys
import token
import tokenize

import astor
import graphviz as gv

__all__ = ["Block", "Link", "CFG"]


class Block(object):
    """
    Basic block in a control flow graph.

    Contains a list of statements executed in a program without any control
    jumps. A block of statements is exited through one of its exits. Exits are
    a list of Links that represent control flow jumps.
    """

    __slots__ = ["id", "statements", "func_calls", "predecessors", "exits", "related_variables"]

    def __init__(self, id):
        # Id of the block.
        self.id = id
        # Statements in the block.
        self.statements = []
        # Calls to functions inside the block (represents context switches to
        # some functions' CFGs).
        self.func_calls = []
        # Links to predecessors in a control flow graph.
        self.predecessors = []
        # Links to the next blocks in a control flow graph.
        self.exits = []
        self.related_variables = set()

    def __del__(self):
        self.statements.clear()
        self.func_calls.clear()
        self.predecessors.clear()
        self.exits.clear()

    """
        取得块号和该块的第一个语句所在的行数
    """

    def __str__(self):
        if self.statements:
            return "block:{}@{}".format(self.id, self.at())
        return "empty block:{}".format(self.id)

    """
        打印该block的基本信息，包括从哪里退出，statement有什么
    """

    def __repr__(self):
        txt = "{} with {} exits".format(str(self), len(self.exits))
        if self.statements:
            txt += ", body=["
            txt += ", ".join([ast.dump(node) for node in self.statements])
            txt += "]"
        return txt

    def at(self):
        """
        Get the line number of the first statement of the block in the program.
        """
        if self.statements and self.statements[0].lineno >= 0:
            return self.statements[0].lineno
        return None

    def is_empty(self):
        """
        Check if the block is empty.
        Returns:
            A boolean indicating if the block is empty (True) or not (False).
        """
        return len(self.statements) == 0

    """
    def strip_comment(self, src):
        clean_src = ""

        prev_toktype = token.INDENT
        first_line = None
        last_lineno = -1
        last_col = 0

        tokgen = tokenize.generate_tokens(src)
        for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokgen:
            if 0:   # Change to if 1 to see the tokens fly by.
                print("%10s %-14s %-20r %r" % (
                    tokenize.tok_name.get(toktype, toktype),
                    "%d.%d-%d.%d" % (slineno, scol, elineno, ecol),
                    ttext, ltext
                    ))
            if slineno > last_lineno:
                last_col = 0
            if scol > last_col:
                mod.write(" " * (scol - last_col))
            if toktype == token.STRING and prev_toktype == token.INDENT:
                # Docstring
                mod.write("#--")
            elif toktype == tokenize.COMMENT:
                # Comment
                mod.write("##\n")
            else:
                mod.write(ttext)
            prev_toktype = toktype
            last_col = ecol
            last_lineno = elineno
        """

    def get_source(self):
        """
        Get a string containing the Python source code corresponding to the
        statements in the block.
        Returns:
            A string containing the source code of the statements.
        """
        src = "#" + str(self.id) + '\n'

        def get_source_helper(source):
            idx, p = 0, 0
            while p != 0 or idx == 0 or source[idx] != ':':
                if source[idx] in "\"'":
                    ridx = source[idx + 1:].index(source[idx]) + idx + 1
                    while True:
                        try:
                            eval(source[idx: ridx + 1].replace("\n", ''))
                            break
                        except:
                            ridx += 1
                    idx = ridx + 1
                    continue
                if source[idx] in '([{':
                    p += 1
                elif source[idx] in ')]}':
                    p -= 1
                idx += 1
            source = source[: idx + 1].split('\n')
            source = ''.join(map(lambda s: s.strip(), source))
            return source + "\n"

        for statement in self.statements:
            source = astor.to_source(statement)
            if type(statement) in [ast.If, ast.For, ast.While, ast.With, ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef]:
                src += get_source_helper(source)
            elif type(statement) == ast.Try:
                src += (astor.to_source(statement)).split('\n')[0] + "\n"
            else:
                src += astor.to_source(statement)
        return src
    """
        获得相关变量所在语句标红的block的语句
    """
    def get_red_source(self, all_variable):
        """
        Get a string containing the Python source code corresponding to the
        statements in the block.
        Returns:
            A string containing the source code of the statements.
        """
        src = "#" + str(self.id) + '\n'

        def get_source_helper(source):
            idx, p = 0, 0
            while p != 0 or idx == 0 or source[idx] != ':':
                if source[idx] in "\"'":
                    ridx = source[idx + 1:].index(source[idx]) + idx + 1
                    while True:
                        try:
                            eval(source[idx: ridx + 1].replace("\n", ''))
                            break
                        except:
                            ridx += 1
                    idx = ridx + 1
                    continue
                if source[idx] in '([{':
                    p += 1
                elif source[idx] in ')]}':
                    p -= 1
                idx += 1
            source = source[: idx + 1].split('\n')
            source = ''.join(map(lambda s: s.strip(), source))
            return source + "\n"

        for statement in self.statements:
            source = astor.to_source(statement)
            flag = False
            if type(statement) in [ast.If, ast.For, ast.While, ast.With, ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef]:
                cur_statement_content = get_source_helper(source)
                cur_statement_content = cur_statement_content.replace('<', '&lt;')
                cur_statement_content = cur_statement_content.replace('>', '&gt;')
                for v in all_variable:
                    if v in cur_statement_content:
                        flag = True
                        break
                if flag:
                    src += '<FONT COLOR="red">' + cur_statement_content + '</FONT>'
                else:
                    src += cur_statement_content
            else:
                cur_statement_content = astor.to_source(statement)
                cur_statement_content = cur_statement_content.replace('<', '&lt;')
                cur_statement_content = cur_statement_content.replace('>', '&gt;')
                for v in all_variable:
                    if v in cur_statement_content:
                        flag = True
                        break
                if type(statement) == ast.Try:
                    if flag:
                        src += '<FONT COLOR="red">' + (astor.to_source(statement)).split('\n')[0] + '</FONT>'
                    else:
                        src += (astor.to_source(statement)).split('\n')[0]
                else:
                    if flag:
                        src += '<FONT COLOR="red">' + astor.to_source(statement) + '</FONT>'
                    else:
                        src += astor.to_source(statement)
        label = '<'+src+'>'
        label = label.replace('\n', '<br/>')
        return label

    def get_calls(self):
        """
        Get a string containing the calls to other functions inside the block.

        Returns:
            A string containing the names of the functions called inside the
            block.
        """
        txt = ""
        for func_call_entry in self.func_calls:
            txt += func_call_entry["name"] + "\n"
        return txt


class Link(object):
    """
    Link between blocks in a control flow graph.

    Represents a control flow jump between two blocks. Contains an exitcase in
    the form of an expression, representing the case in which the associated
    control jump is made.
    """

    __slots__ = ["source", "target", "exitcase"]

    def __init__(self, source, target, exitcase=None):
        assert type(source) == Block, "Source of a link must be a block"
        assert type(target) == Block, "Target of a link must be a block"
        # Block from which the control flow jump was made.
        self.source = source
        # Target block of the control flow jump.
        self.target = target
        # 'Case' leading to a control flow jump through this link.
        self.exitcase = exitcase

    def __str__(self):
        return "link from {} to {}".format(str(self.source), str(self.target))

    def __repr__(self):
        if self.exitcase is not None:
            return "{}, with exitcase {}".format(str(self), ast.dump(self.exitcase))
        return str(self)

    def get_exitcase(self):
        """
        Get a string containing the Python source code corresponding to the
        exitcase of the Link.

        Returns:
            A string containing the source code.
        """
        if self.exitcase:
            return astor.to_source(self.exitcase)
        return ""

    def __del__(self):
        self.source = None
        # Target block of the control flow jump.
        self.target = None
        # 'Case' leading to a control flow jump through this link.
        self.exitcase = None


class CFG(object):
    """
    Control flow graph (CFG).

    A control flow graph is composed of basic blocks and links between them
    representing control flow jumps. It has a unique entry block and several
    possible 'final' blocks (blocks with no exits representing the end of the
    CFG).
    """

    def __init__(self, name, asynchr=False):
        """
        The constructor of CFG class. Only name of this graph is required.
        """
        assert type(name) == str, "Name of a CFG must be a string"
        assert type(asynchr) == bool, "Async must be a boolean value"
        # Name of the function or module being represented.
        self.name = name
        # Type of function represented by the CFG (sync or async). A Python
        # program is considered as a synchronous function (main).
        self.asynchr = asynchr
        # Entry block of the CFG.
        self.entryblock = None
        # Final blocks of the CFG.
        self.finalblocks = []
        # Sub-CFGs for functions defined inside the current CFG.
        self.functioncfgs = {}
        self.class_cfgs = {}

        self.function_args = {}
        self.class_args = {}

        """
            自己加的
            存储每一个子function的定义代码在总的代码中的第一行
            key为子函数名，value为行数
        """
        self.function_firstRow = {}

        self.function_define_statement = {}

        self.function_related_Parameters = {}

        """
            自己加的
            存储每一个子class的定义代码在总的代码中的第一行
            key为子类名，value为行数
        """
        self.class_firstRow = {}

        """
            自己加的
            里面存储主函数中调用子函数的行
            key为函数名，value为一个list，里面存储调用该子函数的行
        """
        # FIXME  function_callRow
        self.function_callRow = {}

        """
            自己加的
            里面存储主函数中使用子类的行
            key为函数名，value为一个list，里面存储使用该子类的行
        """
        self.class_callRow = {}

        """
        自己加的，表示所有与load_model有关的变量
        """
        self.all_variable = {"load_model"}

    def __del__(self):
        pass

    def __str__(self):
        return "CFG for {}".format(self.name)

    def remove_comments(self, src):
        pass

    def get_all_blocks(self):
        """
        Get a list of code blocks in this CFG; This is generated by BFS order.

        Returns:
            A list of code blocks.
        """
        import queue

        all_blocks = []

        visited = set()
        working_queue = queue.Queue()
        working_queue.put(self.entryblock)

        while not working_queue.empty():
            block = working_queue.get()
            # this block has been visited
            if block.id in visited:
                continue
            all_blocks.append(block)
            visited.add(block.id)
            for suc_link in block.exits:
                if suc_link.target.id not in visited:
                    working_queue.put(suc_link.target)
        return all_blocks
        # def dfs(start_block):
        #    # non-recurisve implementation of DFS search

    def __iter__(self):
        """
        Generator that yields all the blocks in the current graph, then
        recursively yields from any sub graphs
        """
        visited = set()
        to_visit = [self.entryblock]

        while to_visit:
            block = to_visit.pop(0)
            visited.add(block)
            for exit_ in block.exits:
                if exit_.target in visited or exit_.target in to_visit:
                    continue
                to_visit.append(exit_.target)
            yield block

        for subcfg in self.functioncfgs.values():
            yield from subcfg



    """
        自己加的
        用来返回给定的tuple中所有的ast.Name类的id
    """

    def findVarInTuple(tuple):
        res = {}

        def helper():
            for ele in tuple.elts:
                if isinstance(ele, ast.Name):
                    res.add(ele.id)
                if isinstance(ele, ast.Tuple):
                    helper(ele)

        helper()
        return res

    """
        建CFG图
    """

    def _visit_blocks(self, graph, block, visited=[], calls=True):
        # Don't visit blocks twice.
        if block.id in visited:
            return

        # FIXME 判断该Block中是否有与深度神经网络（“loadmodel”）相关的变量，将该变量加入到self.all_variable中，并将该block标红
        nodelabel = block.get_source()
        # print("Node_LABEL", " ", nodelabel)
        # nodelabel = '<font color="red">This part is red</font>' + nodelabel
        # print("NODE_LABEL______________________")
        # print(nodelabel)
        # print("___________________________________________")

        flag = False  # 存储是否将该block标红的遍历
        # cur_statements = []
        # is_red = False

        # 存储所有与load_model有关的变量名的list:self.all_variable
        for statement in block.statements:
            target_var_list = set()
            content = ast.dump(statement)

            if isinstance(statement, ast.If):
                content = ast.dump(statement.test)

            # 遍历self.all_variable，如果该statement中有与statement中包含与load_model相关的变量，就把该block标红
            for v in self.all_variable:
                if v in content:
                    print("fffirst", content)
                    flag = True  # 如果该statement中有与statement中包含与load_model相关的变量,就把flag变为true，表示该块应该标红
                    block.related_variables.add(v)

                    # 如果该statement是赋值语句，就把等号左边的变量全部都加入到all_variable中
                    if isinstance(statement, ast.Assign):
                        for target in statement.targets:
                            if isinstance(target, ast.Tuple):
                                tuple_val = self.findVarInTuple(target)
                                target_var_list = target_var_list | tuple_val
                                #将该block的相关变量加入到block.related_variables集合中
                                block.related_variables = block.related_variables | tuple_val
                            elif isinstance(target, ast.Name):
                                target_var_list.add(target.id)
                                block.related_variables.add(target.id)

            self.all_variable = self.all_variable | target_var_list

        state = "\n".join([ast.dump(node) for node in block.statements])
        print("state:" + state)

        print(self.all_variable)
        print(flag)

        if flag:
            nodelabel = block.get_red_source(self.all_variable)
            print("RED_LABEL")
            print(nodelabel)
            graph.node(str(block.id), label=nodelabel, color='red')
        else:
            graph.node(str(block.id), label=nodelabel)

        # FIXME 对一个子函数进行调用时，如果给它传递的参数是相关参数，那么就将该函数定义中对应的形参存储起来，存储到一个字典里，在构建子函数的CFG时，将其传递过去
        for statement in block.statements:
            if hasattr(statement, 'value'):
                if isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name):  # 如果该语句是方法调用语句
                    func_name = statement.value.func.id  # 获取调用的函数名
                    if func_name in self.function_define_statement.keys():  # 如果这个函数是子函数，不是print之类的
                        curFuncRelate = set()
                        # 获取参数
                        flag = False
                        args = []  # 函数调用所用的参数
                        for arg in statement.value.args:
                            args.append(arg.id)

                        define_statement = self.function_define_statement[func_name]
                        params = []  # 函数定义中的形参
                        for arg in define_statement.args.args:
                            params.append(arg.arg)

                        for i in range(len(args)):
                            if args[i] in self.all_variable:
                                curFuncRelate.add(params[i])
                        if func_name in self.function_related_Parameters.keys():
                            self.function_related_Parameters[func_name] |= curFuncRelate
                        else:
                            self.function_related_Parameters[func_name] = curFuncRelate

        visited.append(block.id)
        # Show the block's function calls in a node.
        if calls and block.func_calls:
            calls_node = str(block.id) + "_calls"
            calls_label = block.get_calls().strip()
            graph.node(calls_node, label=calls_label, _attributes={"shape": "box"})
            graph.edge(
                str(block.id),
                calls_node,
                label="calls",
                _attributes={"style": "dashed"},
            )
        # Recursively visit all the blocks of the CFG.
        for exit in block.exits:
            self._visit_blocks(graph, exit.target, visited, calls=calls)
            edgelabel = exit.get_exitcase().strip()
            graph.edge(str(block.id), str(exit.target.id), label=edgelabel)

    def _build_visual(self, format="pdf", calls=True):
        graph = gv.Digraph(
            name="cluster" + self.name, format=format, graph_attr={"label": self.name}
        )
        self._visit_blocks(graph, self.entryblock, visited=[], calls=False)
        return graph

    def build_visual(self, format, calls=True, show=True):
        """
        Build a visualisation of the CFG with graphviz and output it in a DOT
        file.

        Args:
            filename: The name of the output file in which the visualisation
                      must be saved.
            format: The format to use for the output file (PDF, ...).
            show: A boolean indicating whether to automatically open the output
                  file after building the visualisation.
        """
        graph = self._build_visual(format, calls)
        print("GRAPH", graph)
        return graph

    def flatten(self):
        flattend_cfg = {}

        def process_cfg(cfg, dotted_name=["mod"], name_type="mod"):
            fully_qualified_name = ".".join(dotted_name)
            flattend_cfg[fully_qualified_name] = cfg
            for fun_name_tup, fun_cfg in cfg.functioncfgs.items():
                process_cfg(
                    fun_cfg,
                    dotted_name=dotted_name + [fun_name_tup[1]],
                    name_type="func",
                )

            for cls_name, cls_cfg in cfg.class_cfgs.items():
                process_cfg(
                    cls_cfg, dotted_name=dotted_name + [cls_name], name_type="cls"
                )

        process_cfg(self)

        return flattend_cfg

    # """
    #     自己加的
    #     获取每一个子function的定义代码在总的代码中的第一行
    # """
    # def getFucFirstRow(self):
    #     for (block_id, fun_name), fun_cfg in self.functioncfgs.items():
    #         line = fun_cfg.get_all_blocks()[0].at()
    #         self.function_firstRow[fun_name] = line
    #
    # """
    #     自己加的
    #     获取每一个子class的定义代码在总的代码中的第一行
    # """
    # def getClassFirstRow(self):
    #     for class_name, class_cfg in self.class_cfgs.items():
    #         line = class_cfg.get_all_blocks()[0].at()
    #         self.class_firstRow[class_name] = line

    # FIXME 获取每一个子函数的定义代码在总的代码中的第一行，以及在该程序中调用该子函数的地方
    def getFuctionCallRow(self):
        # 首先遍历获取所有的function
        for (block_id, fun_name), fun_cfg in self.functioncfgs.items():
            currows = []
            # 之后遍历该CFG中所有的块，判断该块的所有的statement中是否有这个function的调用
            for block in self.get_all_blocks():
                # 判断该块的所有的statement中是否有这个function的调用
                for statement in block.statements:
                    content = ast.dump(statement)
                    fun_str = "id='" + fun_name + "'"
                    if not isinstance(statement, ast.FunctionDef) and hasattr(statement, 'value') \
                            and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) \
                            and statement.value.func.id == fun_name:
                        currows.append(statement.lineno)
                    if isinstance(statement, ast.FunctionDef) and statement.name == fun_name:
                        self.function_define_statement[fun_name] = statement
                        self.function_firstRow[fun_name] = statement.lineno

            self.function_callRow[fun_name] = currows

    # FIXME 获取每一个子类的定义代码在总的代码中的第一行，以及在该程序中调用该子类的地方
    def getClassCallRow(self):
        for class_name, class_cfg in self.class_cfgs.items():
            currows = []
            # 之后遍历该CFG中所有的块，判断该块的所有的statement中是否有这个class的调用
            for block in self.get_all_blocks():
                # 判断该块的所有的statement中是否有这个class的调用
                for statement in block.statements:
                    content = ast.dump(statement)
                    print(type(content))
                    class_str = "id='" + class_name + "'"
                    if not isinstance(statement, ast.ClassDef) and hasattr(statement, 'value') \
                            and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) \
                            and statement.value.func.id == class_name:
                        currows.append(statement.lineno)
                    if isinstance(statement, ast.ClassDef) and statement.name == class_name:
                        self.class_firstRow[class_name] = statement.lineno

            self.function_callRow[class_name] = currows
