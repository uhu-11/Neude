import os
import pickle
import random
import sys
import time
import sys
from datetime import datetime

import numpy as np
import psutil
import hashlib
import logging
import functools
import multiprocessing as mp

from PIL import Image

from pythonfuzz.Coverages.LineCoverage import LineCoverage

mp.set_start_method('fork')

from pythonfuzz import corpus, tracer
from pythonfuzz.Coverage import Coverage
from pythonfuzz.utils import SaveImgToPath


logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.DEBUG)

# 设置采样窗口为5秒
SAMPLING_WINDOW = 5 # IN SECONDS

try:
    lru_cache = functools.lru_cache
except:
    import functools32
    lru_cache = functools32.lru_cache

'''
这段代码定义了一个多进程工作者进程，该进程接收测试数据并调用目标函数，同时捕获异常并记录覆盖率。
'''

global_coverage = None

def worker(target, child_conn, close_fd_mask):
    # Silence the fuzzee's noise
    class DummyFile:
        """No-op to trash stdout away."""
        def write(self, x):
            pass
    # 捕获警告并将日志级别设置为 ERROR
    logging.captureWarnings(True)

    logging.getLogger().setLevel(logging.ERROR)
    # 根据 close_fd_mask 的值决定是否关闭 stdout 和 stderr
    if close_fd_mask & 1:
        sys.stdout = DummyFile()
    if close_fd_mask & 2:
        sys.stderr = DummyFile()

    # 使用 tracer.trace 作为系统跟踪函数，以监控代码执行路径。
    sys.settrace(tracer.trace)
    '''
    在循环中，从 child_conn 子进程接收字节数据 buf，并将其传递给 target 函数调用。
    如果 target 函数执行过程中出现异常，则捕获异常并记录日志，然后将异常发送回父进程，并终止循环。
    否则，获取代码覆盖率并将其发送回父进程。
    '''
    coverage = LineCoverage(target)
    while True:
        buf = child_conn.recv_bytes()
        buf = pickle.loads(buf)
        # logging.info("child-buf-type:", type(buf))
        try:
            target(*buf.get_params())
        except Exception as e:
            logging.exception(e)
            child_conn.send(e)
            break
        else:
            child_conn.send_bytes(b'%d' % coverage.get_line_coverage())

            #传送覆盖的行数据
            serialized_data = pickle.dumps(coverage.get_coveraged_lines())  # 序列化字典为字节数据
            child_conn.send_bytes(serialized_data)

            child_conn.send_bytes(pickle.dumps(coverage.update()))



class Fuzzer(object):
    def __init__(self,
                 target,   #被测试的目标函数
                 dirs=None,   #存储输入种子文件的目录列表
                 exact_artifact_path=None,   #保存产生的特定输入（例如触发错误的输入）的路径
                 rss_limit_mb=2048,     #内存使用限制，以 MB 为单位
                 timeout=120,            #每次测试的最大时间
                 regression=False,      #是否启用回归测试模式
                 max_input_size=4096,   #输入用例的最大大小
                 close_fd_mask=0,      #关闭文件描述符的掩码
                 runs=-1,            #总的运行次数。
                 dict_path=None):
        self._target = target
        self._dirs = [] if dirs is None else dirs
        self._exact_artifact_path = exact_artifact_path
        self._rss_limit_mb = rss_limit_mb
        self._timeout = timeout
        self._regression = regression
        self._close_fd_mask = close_fd_mask
        self._corpus = corpus.Corpus(self._dirs, max_input_size, dict_path, self._target)
        self._total_executions = 0
        self._executions_in_sample = 0   #当前采样窗口内的执行次数
        self._last_sample_time = time.time()    #上次采样的时间戳
        self._total_coverage = 0
        self._p = None   #存储子进程对象
        self.runs = runs    #fuzz的运行次数

        self._total_lines = {}

    '''
    记录模糊测试的统计信息，包括总执行次数、日志类型、总覆盖率、语料库长度、每秒执行次数和 RSS
        记录日志之后采样窗口就重置了，有两种情况记录日志，一种是生成的输入覆盖率更大了，一种是执行时间到达时间窗口的上限了
    '''
    def log_stats(self, log_type):
        #计算rss内存使用限制，使用 psutil 获取当前进程和子进程的内存使用情况（驻留集大小），并将其转换为 MB
        rss = (psutil.Process(self._p.pid).memory_info().rss + psutil.Process(os.getpid()).memory_info().rss) / 1024 / 1024
        # 计算当前采样窗口每秒执行次数 execs_per_second
        endTime = time.time()
        execs_per_second = int(self._executions_in_sample / (endTime - self._last_sample_time))
        # 重置采样窗口
        self._last_sample_time = time.time()
        self._executions_in_sample = 0
        #使用 logging.info 记录模糊测试的统计信息，包括总执行次数、日志类型、总覆盖率、语料库长度、每秒执行次数和 RSS。
        logging.info('#{} {}     cov: {} corp: {} exec/s: {} rss: {} MB'.format(
            self._total_executions, log_type, self._total_coverage, self._corpus.length, execs_per_second, rss))
        return rss

    def save_image(self, img):
        img = np.array(img, dtype=np.uint8)
        image = Image.fromarray(img)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_number = random.randint(1000, 9999)
        file_name = f"image_{current_time}_{random_number}.png"
        path = "./images/"+file_name
        if not os.path.exists("images"):
            os.makedirs("images")
        image.save(path)
        return path


    '''
    该函数用于在检测到崩溃时，将导致崩溃的输入数据保存到文件中。
    
    具体来说，它计算输入数据的 SHA-256 哈希值，并将其用作文件名的一部分，确保每个输入都有唯一的文件名。
    ##### buf 是引发错误的输入用例
    '''
    def write_sample(self, buf, prefix='crash-'):
        #buf就是seed
        #计算buf的hash值
        m = hashlib.sha256()
        m.update(buf)
        # 如果自己已经设置了保存路径的话，就按照self._exact_artifact_path保存测试用例
        if self._exact_artifact_path:
            crash_path = self._exact_artifact_path
        else:
            dir_path = 'crashes'
            isExist = os.path.exists(dir_path)
            if not isExist:
              os.makedirs(dir_path)
              logging.info("The crashes directory is created")

            crash_path = dir_path + "/" + prefix + m.hexdigest() + '.txt'
        with open(crash_path, 'w') as f:
            # logging.info(type(pickle.loads(buf)))
            params = pickle.loads(buf).get_params()
            for param in params:
                if isinstance(param, list):
                    path = self.save_image(param)
                    f.write(f'{path}\n')
                else:
                    f.write(f'{str(param)}\n')
        logging.info('sample was written to {}'.format(crash_path))
        if len(buf) < 200:
            logging.info('sample = {}'.format(buf.hex()))

    def start(self):
        logging.info("#0 READ units: {}".format(self._corpus.length))
        # self.log_stats("hhhhhhh#0 READ unit")
        exit_code = 0
        parent_conn, child_conn = mp.Pipe()
        # 创建一个子进程，执行worker函数
        self._p = mp.Process(target=worker, args=(self._target, child_conn, self._close_fd_mask))
        self._p.start()

        #进入主循环，持续进行模糊测试
        while True:
            # 如果设置了运行次数限制(self.runs != -1)，且总执行次数达到或超过限制，终止子进程并退出循环
            if self.runs != -1 and self._total_executions >= self.runs:
                self._p.terminate()
                logging.info('did %d runs, stopping now.', self.runs)
                break
            # 使用corpus生成测试输入generate_input
            buf = self._corpus.generate_input()
            # print(f'buf的类型{type(buf)}, buf的shape{np.array(buf).shape}')
            #通过管道 parent_conn 将输入发送给子进程
            # flattened_list = [item for sub1 in buf for sub2 in sub1 for item in sub2]
            parent_conn.send_bytes(pickle.dumps(buf))
            # 等待子进程在指定的超时时间内响应。
            # 如果超时未响应，终止子进程，记录日志，并将超时导致的输入保存到文件中，退出循环。
            if not parent_conn.poll(self._timeout):
                self._p.kill()
                logging.info("=================================================================")
                logging.info("timeout reached. testcase took: {}".format(self._timeout))
                self.write_sample(pickle.dumps(buf), prefix='timeout-')
                break

            try:
                total_coverage = int(parent_conn.recv_bytes())
                serialized_data = parent_conn.recv_bytes()
                coveraged_lines = pickle.loads(serialized_data)

                has_new_coverage = pickle.loads(parent_conn.recv_bytes())

            except ValueError as e:
                self.write_sample(pickle.dumps(buf))
                exit_code = 76
                break
            #更新总执行次数 self._total_executions 和当前采样窗口内的执行次数 self._executions_in_sample。
            self._total_executions += 1
            self._executions_in_sample += 1
            rss = 0
            '''
            如果新覆盖率 total_coverage 超过当前覆盖率 self._total_coverage，记录 "NEW" 日志，更新总覆盖率，并将新的输入添加到语料库。
            如果一次生成中，生成的测试用例所达覆盖率最大，则当前采样窗口结束
            '''
            if total_coverage > self._total_coverage:
                rss = self.log_stats("NEW")
                self._total_coverage = total_coverage
                self._corpus.put(buf)
                #保存种子图片到本地
                SaveImgToPath.save(buf)
            elif has_new_coverage:
                rss = self.log_stats("NEW")
                self._corpus.put(buf)
                # 保存种子图片到本地
                SaveImgToPath.save(buf)

            else:
                '''
                    否则，如果自上次采样以来时间超过了采样窗口 SAMPLING_WINDOW，记录 "PULSE" 日志。
                '''
                if (time.time() - self._last_sample_time) > SAMPLING_WINDOW:
                    rss = self.log_stats('PULSE')

            # merged, more = merge_dicts_and_diff(self._total_lines, coveraged_lines)
            # self._total_lines = merged
            # if len(more) > 0:
            #     rss = self.log_stats("MORE_LINES")




            '''
            如果 RSS（驻留集大小）超过了内存限制 self._rss_limit_mb，记录日志，并将当前输入保存到文件中，终止子进程，退出循环。
            '''
            if rss > self._rss_limit_mb:
                logging.info('MEMORY OOM: exceeded {} MB. Killing worker'.format(self._rss_limit_mb))
                self.write_sample(pickle.dumps(buf))
                self._p.kill()
                break
        '''
        等待子进程终止。
        '''
        self._p.join()
        #退出
        sys.exit(exit_code)


# def merge_dicts_and_diff(dict1, dict2):
#     merged_dict = {}
#     added_keys = {}
#
#     # 遍历 dict1 的键值对
#     for key, value1 in dict1.items():
#         if key in dict2:
#             value2 = dict2[key]
#             # 合并并去重
#             merged_value = list(set(value1 + value2))
#             merged_dict[key] = merged_value
#             added_tuples = [t for t in value2 if t not in value1]  # 找出新增的元组
#             if added_tuples:
#                 added_keys[key] = added_tuples  # 记录新增的键和元组
#         else:
#             merged_dict[key] = value1
#
#     # 处理 dict2 中存在而 dict1 中不存在的键值对
#     for key, value2 in dict2.items():
#         if key not in dict1:
#             merged_dict[key] = value2
#             added_keys[key] = value2  # 记录新增的键和元组
#
#     return merged_dict, added_keys
#
# # 计算字典中所有值中元组的总数量
# def count_tuples_in_dict(d):
#     count = 0
#     for key, value in d.items():
#         count += len(value)
#     return count