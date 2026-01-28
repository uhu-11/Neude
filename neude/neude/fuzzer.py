import inspect
import os
import pickle
import random
import sys
import time
from PIL import Image
from datetime import datetime
import shutil
import numpy as np
import psutil
import json
import hashlib
import subprocess
import typing
import logging
import signal
import functools
import multiprocessing as mp
import prettytable
# from sklearn.metrics.pairwise import cosine_similarity
import copy
import coverage as cv
import glob
import tensorflow as tf
import traj_dist.distance as tdist

from neude.config import PREDICT_ERROR_SEED_PATH,COV_HTML_PATH, COV_REPORT_PATH, SEED_SAVE_PATH, ERROR_INFOS_DIR
from neude.config import CRASH_SEED_PATH,DATA_SAVE_PATH,PROJECT_ERROR_LOG, ITER_COV_REPORT_PATH,OB_TL_PREDICTIONS_NPYS
from neude.config import CONTROL_PREDICTIONS_NPYS, PLANNING_PREDICTIONS_NPYS
from neude.config import USE_FUNCTIONS
try:
    from neude.config import NEUDE_METHOD
except ImportError:
    NEUDE_METHOD = None


from neude.Coverages.LineCoverage import LineCoverage
from neude.Coverages.PTCoverage import PTCoverage
from neude.Coverages import PNAC
from neude.Coverages.NacCoverage import NacCoverage
from selection_method.necov_method.neural_cov import CovInit
from .utils import ParamsType, cal_iou, plt, cal_tdist
from . import Seed, config

mp.set_start_method('fork')

from neude import corpus, tracer
from neude.Coverage import Coverage
from neude.utils import SaveUtil, use_predict, htmlpaser
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'pylot'))
from pylot.global_var import COV_FILE_PATH


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

global_has_model = False
global_nc_cov = False
global_coverage = None
global_bacth_size=1
global_timeout=0


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
    '''
    在循环中，从 child_conn 子进程接收字节数据 buf，并将其传递给 target 函数调用。
    如果 target 函数执行过程中出现异常，则捕获异常并记录日志，然后将异常发送回父进程，并终止循环。
    否则，获取代码覆盖率并将其发送回父进程。
    '''
    ptCoverage = PTCoverage()
    lineCoverage = LineCoverage()
    pnac = PNAC.PNAC()

    def runner(buf, que):
        tf.compat.v1.enable_eager_execution()
        res = target(**buf.get_params())
        que.put(res)
        que.close()  
        que.join_thread()
    model_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/traffic_light_detection/frozen_graph.pb'
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')  
    # tf.config.experimental.set_visible_devices([physical_devices[1]], 'GPU')
    # tf.config.experimental.set_memory_growth([physical_devices[1]], True)
    # if physical_devices:
    #     try:
    #         # 将所有 GPU 设置为可见
    #         tf.config.experimental.set_visible_devices(physical_devices, 'GPU')
    #         # 对每个 GPU 启用内存增长
    #         # for gpu in physical_devices:
    #         #     tf.config.experimental.set_memory_growth(gpu, True)
    #         print("所有 GPU 已设置为可见，并启用了内存增长。")
    #     except RuntimeError as e:
    #         # GPU 已被初始化，设置会失败
    #         print("设置 GPU 配置时出错：", e)
    #     else:
    #         print("检测到 GPU 设备。")
    i=0
    while True:
        i+=1
        np.save('/media/lzq/D/lzq/pylot_test/img_ind.npy', np.array(0))

        buf = pickle.loads(child_conn.recv_bytes())
        y_predict_pool = pickle.loads(child_conn.recv_bytes())
        y_pool = pickle.loads(child_conn.recv_bytes())
        executions = pickle.loads(child_conn.recv_bytes())
        PNAC.global_pnac_map = pickle.loads(child_conn.recv_bytes())
        htmlpaser.global_line_status_vector_map = pickle.loads(child_conn.recv_bytes())
        htmlpaser.global_keys = pickle.loads(child_conn.recv_bytes())

   
        
        # logging.info("child-buf-type:", type(buf))
        is_predict_label_equal_groundtruth = False
        cov = cv.Coverage(data_file=f'{COV_FILE_PATH}/.coverage.main.{mp.current_process().pid}.main', branch=True)
        try:
            sys.settrace(tracer.trace)
            # param_names = list(inspect.signature(target).parameters.keys())
            # param_values = []
            # for i in range(len(buf.get_params().items())):
            #     v = buf.get_params().get(param_names[i])
            #     if v:
            #         param_values.append(v)
                    
            que = mp.Queue()
            target_process = mp.Process(target=runner, args=(buf, que,))
            cov.start()
            # cov.exclude('@coverage_decorator')
            # cov.exclude('def coverage_decorator')

            carla_process = subprocess.Popen(
                "export PYLOT_HOME=/media/lzq/D/lzq/pylot_test/pylot/ && "
                "export CARLA_HOME=$PYLOT_HOME/dependencies/CARLA_0.9.10.1/ && "
                "/media/lzq/D/lzq/pylot_test/pylot/scripts/run_simulator.sh",
                shell=True, executable="/bin/bash"
            )

            # uses_predict = use_predict.function_uses_predict(target)
            if global_has_model:

                #print("buf.get_params()", buf.get_params())
                if not global_nc_cov:
                    y_predict = target(**buf.get_params())
                    # print("y_predict", y_predict)
                    int_y = int(buf.get_params().get("y"))
                    y = np.array([int_y])
                    # 获取预测结果,判断预测结果是否正确
                    predict_label = np.argmax(y_predict[0])
                    if(int_y == predict_label):
                        is_predict_label_equal_groundtruth = True
                    
                
                    if y_predict_pool is None:
                        y_predict_pool = y_predict
                        y_pool = y
                    else:
                        y_predict_pool = np.concatenate((y_predict_pool, y_predict), axis=0)
                        y_pool = np.concatenate((y_pool, y), axis=0)
                    
                    #print("fuzzer-y_pool", y_pool)
                    pt_rate, region_length, _ = ptCoverage.get_coverage(y_predict_pool, y_pool, 10, 4)
                    _, _, pt_vector = ptCoverage.get_coverage(y_predict, y, 10, 4)
                    pt_rate = pt_rate*10
                else:
                    print("target_process start..........")
                    target_process.start()
                    start_time = time.time()
    
                    while True:
                        # 检查进程是否已结束
                        # print('检查',target_process.is_alive())
                        target_process.join(1)
                        if not que.empty():
                            x = que.get()
                        if not target_process.is_alive():
                            print('结束')
                            break
                        
                        # 如果超时，则终止进程
                        if time.time() - start_time > 400:
                            print("子进程运行超时，正在终止...")
                            os.kill(target_process.pid, signal.SIGINT)
                            # if os.path.exists(PROJECT_ERROR_LOG) :
                            #     shutil.copy(PROJECT_ERROR_LOG, f'{ERROR_INFOS_DIR}/{i}.txt')
                           
                            raise TimeoutError('子进程运行超时')
                        time.sleep(1)  # 每秒检查一次
                    # print('检查hhhh',target_process.is_alive())
                    # print(x)
                    if target_process.is_alive():
                        target_process.join()
                        if not que.empty():
                            x = que.get()
                    
                    if os.path.exists(PROJECT_ERROR_LOG) and os.path.getsize(PROJECT_ERROR_LOG)>0:
                        print(PROJECT_ERROR_LOG, "exits")
                        with open(PROJECT_ERROR_LOG, 'r') as f:
                            error_content = f.read()

                        os.remove(PROJECT_ERROR_LOG)
                        raise ValueError(error_content)
                    # print('检查llll',target_process.is_alive())
                    '''
                        计算NAC覆盖
                    '''
                    # model_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/traffic_light_detection/frozen_graph.pb'
                    x_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/obstacle_detection/x.npy'
                    # vec_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/obstacle_detection/vec.json'
                    np.save(x_path ,x)
                    # subprocess.run(['python', '/media/lzq/D/lzq/pylot_test/pythonfuzz/pythonfuzz/Coverages/NacCoverage.py', '--model_path', model_path, '--data_path', x_path, '--vec_path', vec_path])
                    # with open(vec_path, 'r') as f:
                    #     cur_seed_nac_vec = json.load(f)
                    # cur_seed_nac_rate = sum(cur_seed_nac_vec)/float(len(cur_seed_nac_vec))

                    #判断被测系统预测的结果是否正确
                    #红绿灯或障碍物检测模块的输出判断
                    y = buf.get_params().get("y")
                    ious = cal_iou.get_set_iou(OB_TL_PREDICTIONS_NPYS, y)
                    answers_ob_tl = [1 if value > 0.5 else 0 for value in ious]
                    print('iou in fuzzer:',ious)
                    print('answers_ob_tl:',answers_ob_tl)
                    planning_label = buf.get_params().get("planning_label")
                    tdists = cal_tdist.get_set_tdist(PLANNING_PREDICTIONS_NPYS, planning_label, i, buf.get_params().get("planning_type_enum"))
                    # print('tdists:',tdists)
                    answers_planning = [1 if value < 0.5 else 0 for value in tdists]
                    

                    control_label = buf.get_params().get("control_label")
                    steer_diff = cal_tdist.get_set_steer_diff(CONTROL_PREDICTIONS_NPYS, control_label, i, buf.get_params().get("planning_type_enum"))
                    answers_control = [1 if value < 0.5 else 0 for value in steer_diff]
                    answers = [
                        1 if answers_ob_tl[i] == 1 and answers_planning[i] == 1 and answers_control[i] == 1 else 0
                        for i in range(len(answers_ob_tl))
                    ]
                    if answers.count(0)==0:
                        is_predict_label_equal_groundtruth = True
                  
                    # print('iou:',iou)
                    # cur_seed_nac_vec = [item for array in cur_seed_nac_vec for item in array]

            else:
                # physical_devices = tf.config.experimental.list_physical_devices('GPU')
                # tf.config.experimental.set_memory_growth(
                #     physical_devices[0], True)
                target(**buf.get_params())



                # nc = NacCoverage(tf_model_path=model_path, x_test=x)
                # # import gc
                # cur_seed_nac_score, cur_seed_nac_vec = nc.get_coverage()
                # tf.keras.backend.clear_session()  # 清除当前的 Keras 会话
                # gc.collect()  

                # tf.compat.v1.reset_default_graph()
                # del nc 
                # cur_seed_nac_vec = [item for array in cur_seed_nac_vec for item in array]
                # print("cur_seed_nac_score:", cur_seed_nac_score)
                
                # target(**buf.get_params())
            # get_recursive_coverage(target, cov)
            cov.stop()
            # 结果保存
            cov.save()
        
            
            
            pnac_map, pnac_rate, cur_seed_nac_vec, cur_seed_nac_rate,total_line_rate, total_1_count, total_1_and_0_count, cur_seed_line_vector, cur_seed_line_rate, batch_nac_vectors =pnac.get_PNAC(global_bacth_size)
            print('用PNAC计算的line rate:', total_line_rate)
            coverage_files = glob.glob(f'{COV_FILE_PATH}/.coverage*')
            for file_path in coverage_files:
                new_file_path = os.path.join(ITER_COV_REPORT_PATH, os.path.basename(file_path))
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                shutil.copy(file_path, new_file_path)

            # 逐个合并文件，跳过不兼容的文件（避免混合分支数据和语句数据）
            for file_path in coverage_files:
                try:
                    cov.combine(data_paths=[file_path], keep=True)
                except Exception as e:
                    # 如果文件不兼容（例如：包含分支数据的文件与不包含分支数据的文件混合）
                    # 跳过该文件并继续处理其他文件
                    logging.warning(f"Skipping incompatible coverage file: {file_path}, error: {e}")
                    continue
            if not os.path.exists(COV_HTML_PATH+"/"+str(executions)):
                os.makedirs(COV_HTML_PATH+"/"+str(executions))
            if not os.path.exists(COV_REPORT_PATH+"/"+str(executions)):
                os.makedirs(COV_REPORT_PATH+"/"+str(executions))
        
            cov.html_report(directory=COV_HTML_PATH+"/"+str(executions))
            # 命令行模式展示结果
            with open(COV_REPORT_PATH+"/"+str(executions)+"/coverage_report_"+str(executions)+".txt", "w") as file:
                cov.report(file=file)
            
            # 提取分支覆盖信息
            # 使用 report() 方法获取摘要信息，然后解析报告文本
            import io
            import re
            total_branch_covered = 0
            total_branches = 0
            total_branch_rate = 0.0
            
            try:
                # 生成报告到字符串缓冲区
                report_buffer = io.StringIO()
                cov.report(file=report_buffer)
                report_text = report_buffer.getvalue()
                
                lines = report_text.strip().split('\n')
                for line in reversed(lines):
                    if line.startswith('TOTAL') or 'TOTAL' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                total_branches = int(parts[3])  # Branch 列（总分支数）
                                br_part = int(parts[4])  # BrPart 列（部分覆盖的分支数）
                                
                                if len(parts) >= 6:
                                    try:
                                        cover_str = parts[5].rstrip('%')
                                        cover_rate = float(cover_str) / 100.0
                                        total_branch_rate = cover_rate
                                        total_branch_covered = int(total_branches * cover_rate)
                                    except:
                                        total_branch_covered = 0
                                        if total_branches > 0:
                                            total_branch_rate = total_branch_covered / total_branches
                                else:
                                    total_branch_covered = 0
                                    if total_branches > 0:
                                        total_branch_rate = total_branch_covered / total_branches
                                break
                            except (ValueError, IndexError) as e:
                                logging.debug(f"Could not parse TOTAL line: {e}")
                                break
                
                if total_branches == 0:
                    branch_data = cov.get_data()
                    for filename in branch_data.measured_files():
                        try:
                            arcs = branch_data.arcs(filename)
                            if arcs:
                                if isinstance(arcs, dict):
                                    for count in arcs.values():
                                        total_branches += 1
                                        if count > 0:
                                            total_branch_covered += 1
                                elif isinstance(arcs, list):
                                    total_branch_covered += len(arcs)
                        except Exception:
                            continue
                    
                    if total_branches == 0 and total_branch_covered > 0:
                        total_branch_rate = 0.0
                    elif total_branches > 0:
                        total_branch_rate = total_branch_covered / total_branches
                        
            except Exception as e:
                logging.warning(f"Could not extract branch coverage from data: {e}")
                total_branch_rate = 0.0
            
            total_line_rate, total_1_count, total_1_and_0_count, cur_seed_line_vector,line_status_vector_map ,cur_line_status_vector_map, global_keys= \
                lineCoverage.get_line_coverage(COV_HTML_PATH+"/"+str(executions))

            sys.settrace(None)
        except Exception as e:
            logging.exception(e)
            import traceback
            error_message = traceback.format_exc()
            with open(f'{ERROR_INFOS_DIR}/{i}.txt', 'w', encoding='utf-8') as file:
                file.write(error_message)
                        
            terminate_extra_fuzz_pids()
            cov.stop()
            cov.save()
            child_conn.send(e)
            
        else:
            child_conn.send_bytes(b'%d' % lineCoverage.get_line_coverage_num())

            child_conn.send_bytes(pickle.dumps(total_line_rate))
            child_conn.send_bytes(pickle.dumps(total_1_count))
            child_conn.send_bytes(pickle.dumps(total_1_and_0_count))
            child_conn.send_bytes(pickle.dumps(cur_seed_line_vector))
            child_conn.send_bytes(pickle.dumps(cur_seed_line_rate))
            child_conn.send_bytes(pickle.dumps(cur_line_status_vector_map))
            child_conn.send_bytes(pickle.dumps(line_status_vector_map))
            child_conn.send_bytes(pickle.dumps(global_keys))
            # 发送分支覆盖信息
            child_conn.send_bytes(pickle.dumps(total_branch_rate))
            child_conn.send_bytes(pickle.dumps(total_branch_covered))
            child_conn.send_bytes(pickle.dumps(total_branches))
            # child_conn.send_bytes(pickle.dumps(cur_seed_nac_score))
            # child_conn.send_bytes(pickle.dumps(cur_seed_nac_vec))
            # 发送batch_nac_vectors（在所有情况下都发送）
            child_conn.send_bytes(pickle.dumps(batch_nac_vectors))

            if global_has_model:
                if not global_nc_cov:
                    child_conn.send_bytes(pickle.dumps(y_predict_pool))
                    child_conn.send_bytes(pickle.dumps(y_pool))
                    child_conn.send_bytes(pickle.dumps(pt_rate))
                    child_conn.send_bytes(pickle.dumps(region_length))
                    child_conn.send_bytes(pickle.dumps(is_predict_label_equal_groundtruth))
                    child_conn.send_bytes(pickle.dumps(pt_vector[0]))
                else:
                    child_conn.send_bytes(pickle.dumps(cur_seed_nac_rate))
                    child_conn.send_bytes(pickle.dumps(cur_seed_nac_vec))
                    child_conn.send_bytes(pickle.dumps(pnac_rate))
                    child_conn.send_bytes(pickle.dumps(pnac_map))
                    
                    child_conn.send_bytes(pickle.dumps(is_predict_label_equal_groundtruth))
                    child_conn.send_bytes(pickle.dumps(answers_ob_tl))
                    child_conn.send_bytes(pickle.dumps(answers_planning))
                    child_conn.send_bytes(pickle.dumps(answers_control))

                    child_conn.send_bytes(pickle.dumps(ious))
                    child_conn.send_bytes(pickle.dumps(tdists))
                    child_conn.send_bytes(pickle.dumps(steer_diff))
        finally:
            # process = subprocess.Popen(
            #     "pkill --signal 9 -f fuzz_list.py",
            #     shell=True, executable="/bin/bash"
            # )
            if carla_process.poll() is None:
                kill_process_tree(carla_process.pid)
        time.sleep(1)

            



class Fuzzer(object):
    def __init__(self,
                 target,   #被测试的目标函数
                 dirs=None,   #存储输入种子文件的目录列表
                 exact_artifact_path=None,   #保存产生的特定输入（例如触发错误的输入）的路径
                 rss_limit_mb=4096,     #内存使用限制，以 MB 为单位
                 timeout=10000,            #每次测试的最大时间
                 regression=False,      #是否启用回归测试模式
                 max_input_size=4096,   #输入用例的最大大小
                 close_fd_mask=0,      #关闭文件描述符的掩码
                 runs=-1,            #总的运行次数。
                 cov_vec_save_iter=500, #保存覆盖向量的轮数 
                 has_model = False,
                 use_nc = False,
                 batch_size=1,
                 dict_path=None):
        global global_has_model
        global global_nc_cov
        global global_timeout
        global global_bacth_size

        global_has_model = has_model
        global_nc_cov = use_nc
        global_bacth_size = batch_size
        global_timeout = timeout
        self._target = target
        self._dirs = [] if dirs is None else dirs
        self._exact_artifact_path = exact_artifact_path
        self._rss_limit_mb = rss_limit_mb
        self._cov_vec_save_iter = cov_vec_save_iter
        self._timeout = timeout
        self._regression = regression
        self._close_fd_mask = close_fd_mask
        self._corpus = corpus.Corpus(self._dirs, max_input_size, dict_path, self._target)
        
        self._total_executions = 0
        self._executions_in_sample = 0   #当前采样窗口内的执行次数
        self._last_sample_time = time.time()    #上次采样的时间戳
        self._p = None   #存储子进程对象
        self.runs = runs    #fuzz的运行次数
        self.batch_size = batch_size
        self._total_seedpool_size = self._corpus.get_corpus_size()-self.batch_size

        self._error_number = 0
        self._predict_error_number = 0
        self._planning_error_number = 0
        self._control_error_number = 0
        # 总覆盖率 表示从开始测试到当前的全部所达覆盖率
        self._max_line_rate = 0
        self._max_1_count = 0
        self._max_1_and_0_count = 0

        self._cur_line_rate = 0
        self._cur_1_count = 0
        self._cur_1_and_0_count = 0
        
        self._pre_pt_rate = 0
        self._y_predict_pool = None
        self._y_pool = None
        self._pre_region_length = 0
        self._max_pt_rate = 0
        self._max_region_length = 0

        self._total_nac_vector=[]
        self.pre_max_nac_rate=0
        self._max_nac_rate=0
        self._cur_nac_rate=0
        self._max_all_nac_rate=0
        self._total_nac_rate=0

        self._combinedCoverage = 0
        self._max_combinedCoverage = 0

        self._cur_pnac=0
        self._max_pnac=0
        self._pnac_map=None
        
        # 分支覆盖统计
        self._max_branch_rate = 0
        self._cur_branch_rate = 0
        self._total_branch_covered = 0
        self._total_branches = 0

        self._use_functions = USE_FUNCTIONS
        self._neude_method = NEUDE_METHOD



        self._init_seed_vectors = []
        self._mutation_seed_vectors = []

        #实验数据收集
        self.plt_line_cov=[]
        self.plt_nac_cov=[]
        self.plt_combine_cov=[]
        self.plt_pnac_cov=[]
        self.code_errors=[]
        self.model_errors=[]
        self.planning_errors=[]
        self.control_errors=[]
        self.iou=[]
        self.tdist=[]
        self.steer_diff=[]
        self.seed_vectors=[]
        self.seed_parents=[]
        self.cur_seed_errors=[]
        self.pnac_maps = []
        self.line_status_vector_maps=[]
        self.cur_line_status_vector_maps=[]
        self.batch_nac_vectors_list = []  # 保存每个迭代的batch_nac_vectors
        self._tb = prettytable.PrettyTable()
        self._tb.field_names = ["Iteration", "size_of_seedpool", 'parent',"code_error_number", "predict_error_number","planning_error_number","control_error_number", "line_rate", "branch_rate", "pt_rate", "nac_rate", "combined_rate", 'pnac_rate', "max_line_rate", "max_branch_rate", "max_pt_rate", "max_nac_rate", "max_combined_rate", "max_pnac_rate"]
# """         
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         if gpus:
#             try:
#                 for gpu in gpus:
#                     tf.config.experimental.set_memory_growth(gpu, True)
#             except RuntimeError as e:
#                 print(e) """
    '''
    记录模糊测试的统计信息，包括总执行次数、日志类型、总覆盖率、语料库长度、每秒执行次数和 RSS
        记录日志之后采样窗口就重置了，有两种情况记录日志，一种是生成的输入覆盖率更大了，一种是执行时间到达时间窗口的上限了
    '''
    def log_stats(self, log_type, buf):
        #计算rss内存使用限制，使用 psutil 获取当前进程和子进程的内存使用情况（驻留集大小），并将其转换为 MB
        rss = (psutil.Process(self._p.pid).memory_info().rss + psutil.Process(os.getpid()).memory_info().rss) / 1024 / 1024
        # 计算当前采样窗口每秒执行次数 execs_per_second
        endTime = time.time()
        execs_per_second = int(self._executions_in_sample / (endTime - self._last_sample_time))
        # 重置采样窗口
        self._last_sample_time = time.time()
        self._executions_in_sample = 0
        #使用 logging.info 记录模糊测试的统计信息，包括总执行次数、日志类型、总覆盖率、语料库长度、每秒执行次数和 RSS。
        # logging.info('#{} {}    line_rate: {} pt_rate: {} combined_rate: {}  corp: {} exec/s: {} rss: {} MB, size_of_seedpool:{}'.format(
        #   self._total_executions, log_type, self._total_line_rate, self._pre_pt_rate, self._combinedCoverage, self._corpus.length, execs_per_second, rss, len(self._corpus.get_inputs())))
        self._tb.clear_rows()
        self._tb.add_row([
            self._total_executions,
            self._total_seedpool_size,
            buf.from_seed_que[-1],
            self._error_number,
            self._predict_error_number,
            self._planning_error_number,
            self._control_error_number,
            f"{self._cur_line_rate:.4f}",  
            f"{self._cur_branch_rate:.4f}",
            f"{self._pre_pt_rate:.4f}", 
            f"{self._cur_nac_rate:.4f}",      
            f"{self._combinedCoverage:.4f}", 
            f"{self._cur_pnac:.4f}", 
            f"{self._max_line_rate:.4f}",
            f"{self._max_branch_rate:.4f}",
            f"{self._max_pt_rate:.4f}",    
            f"{self._max_nac_rate:.4f}",    
            f"{self._max_combinedCoverage:.4f}",
            f"{self._max_pnac:.4f}"    
        ])
        print(self._tb)
        logging.info("===========================================================================================================")
        return rss

    def save_image(self, img):
        # img = np.array(img, dtype=np.uint8)
        # image = Image.fromarray(img)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_number = random.randint(1000, 9999)
        file_name = f"image_{current_time}_{random_number}.png"
        path = "./images/"+file_name
        if not os.path.exists("images"):
            os.makedirs("images")
        img.save(path)
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

            crash_path = dir_path + "/" + str(self._total_executions) + '_' + prefix + m.hexdigest() + '.txt'
        with open(crash_path, 'w') as f:
            # logging.info(type(pickle.loads(buf)))
            params = pickle.loads(buf).get_params()
            for param_name, value in params.items():
                if isinstance(value, Image.Image):
                    path = self.save_image(value)
                    f.write(f'{param_name}={path}\n')
                else:
                    f.write(f'{param_name}={value}\n')
        logging.info('sample was written to {}'.format(crash_path))
        if len(buf) < 200:
            logging.info('sample = {}'.format(buf.hex()))

    def start(self):
        recreate_folder()
        os.makedirs('/home/lzq/result/datas', exist_ok=True)
        try:
            # 清理旧的 coverage 文件，确保所有文件都包含分支数据
            if os.path.exists(COV_FILE_PATH):
                shutil.rmtree(COV_FILE_PATH)
            os.makedirs(COV_FILE_PATH, exist_ok=True)
            
            if os.path.exists(ERROR_INFOS_DIR):
                shutil.rmtree(ERROR_INFOS_DIR)
            os.makedirs(ERROR_INFOS_DIR, exist_ok=True)

            if os.path.exists(COV_HTML_PATH):
                shutil.rmtree(COV_HTML_PATH)
            if os.path.exists(COV_REPORT_PATH):
                shutil.rmtree(COV_REPORT_PATH)
            logging.info("#0 READ units: {}".format(self._corpus.length))
            # self.log_stats("hhhhhhh#0 READ unit")
            exit_code = 0
            parent_conn, child_conn = mp.Pipe()
            # 创建一个子进程，执行worker函数
            self._p = mp.Process(target=worker, args=(self._target, child_conn, self._close_fd_mask))
            self._p.start()



            #进入主循环，持续进行模糊测试
            while True:
                if os.path.exists(COV_FILE_PATH):
                    shutil.rmtree(COV_FILE_PATH)
                os.makedirs(COV_FILE_PATH, exist_ok=True)
                # 如果设置了运行次数限制(self.runs != -1)，且总执行次数达到或超过限制，终止子进程并退出循环
                if self.runs != -1 and self._total_executions >= self.runs:
                    self._p.terminate()
                    logging.info('did %d runs, stopping now.', self.runs)
                    break
                # 在 iteration=102 时自动停止测试
                if self._total_executions >= 102:
                    self._p.terminate()
                    logging.info('Reached iteration 102, stopping test now.')
                    break
                
                param_types = ParamsType.get_parameter_types(self._target)
                '''
                生成测试用例子
                '''
           
                buf, is_init_seed = self._corpus.generate_input(self.batch_size,self._total_seedpool_size,self._target)
                updates = {'--seed_ind':buf.from_seed_que[0]}
                print(updates)
                update_config(updates=updates, file_path='/media/lzq/D/lzq/pylot_test/pylot/configs/mpc2.conf')          
                            
                

                # print(f'buf的类型{type(buf)}, buf的shape{np.array(buf).shape}')
                #通过管道 parent_conn 将输入发送给子进程
                # flattened_list = [item for sub1 in buf for sub2 in sub1 for item in sub2]
                parent_conn.send_bytes(pickle.dumps(buf))
                parent_conn.send_bytes(pickle.dumps(self._y_predict_pool))
                parent_conn.send_bytes(pickle.dumps(self._y_pool))
                parent_conn.send_bytes(pickle.dumps(self._total_executions))
                parent_conn.send_bytes(pickle.dumps(PNAC.global_pnac_map))
                parent_conn.send_bytes(pickle.dumps(htmlpaser.global_line_status_vector_map))
                parent_conn.send_bytes(pickle.dumps(htmlpaser.global_keys))
                
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
                    print('woker运行结束')
                    '''
                    total_line_rate是从开始到当前的总覆盖率
                    total_1_count是从开始到当前的总覆盖行数
                    total_1_and_0_count是从开始到当前的总覆盖行数和未覆盖行数

                    cur_line_rate是当前采样样例的覆盖率
                    cur_1_count是当前采样样例的覆盖行数
                    cur_1_and_0_count是当前采样样例的覆盖行数和未覆盖行数
                    '''
                    total_line_rate = float(pickle.loads(parent_conn.recv_bytes()))
                    total_1_count = pickle.loads(parent_conn.recv_bytes())
                    total_1_and_0_count = pickle.loads(parent_conn.recv_bytes())
                    cur_seed_line_vector = pickle.loads(parent_conn.recv_bytes())
                    cur_seed_line_rate = pickle.loads(parent_conn.recv_bytes())
                    cur_line_status_vector_map = pickle.loads(parent_conn.recv_bytes())
                    line_status_vector_map = pickle.loads(parent_conn.recv_bytes())
                    global_keys = pickle.loads(parent_conn.recv_bytes())
                    # 接收分支覆盖信息
                    total_branch_rate = float(pickle.loads(parent_conn.recv_bytes()))
                    total_branch_covered = pickle.loads(parent_conn.recv_bytes())
                    total_branches = pickle.loads(parent_conn.recv_bytes())
                    
                    self.line_status_vector_maps.append(line_status_vector_map)
                    self.cur_line_status_vector_maps.append(cur_line_status_vector_map)
               
                    htmlpaser.global_line_status_vector_map = line_status_vector_map
                    htmlpaser.global_keys = global_keys
                    
                    # 更新分支覆盖统计
                    self._cur_branch_rate = total_branch_rate
                    self._total_branch_covered = total_branch_covered
                    self._total_branches = total_branches
                    if total_branch_rate > self._max_branch_rate:
                        self._max_branch_rate = total_branch_rate
                    # cur_seed_nac_rate = pickle.loads(parent_conn.recv_bytes())
                    # self._cur_nac_rate = cur_seed_nac_rate
                    # cur_seed_nac_vector = pickle.loads(parent_conn.recv_bytes())
                    # 接收batch_nac_vectors（在所有情况下都接收）
                    batch_nac_vectors = pickle.loads(parent_conn.recv_bytes())
                    self.batch_nac_vectors_list.append(batch_nac_vectors)

                    self.pre_max_nac_rate = self._max_nac_rate
                    cur_seed_nac_rate = 0
                    pt_rate = 0
                    region_length = 0
                    y_pool, y_predict_pool = None, None
                    if global_has_model:
                        if not global_nc_cov:
                            y_predict_pool = pickle.loads(parent_conn.recv_bytes())
                            y_pool = pickle.loads(parent_conn.recv_bytes())
                            pt_rate = pickle.loads(parent_conn.recv_bytes())
                            region_length = pickle.loads(parent_conn.recv_bytes())
                            is_predict_true = pickle.loads(parent_conn.recv_bytes())
                            cur_seed_pt_vector = pickle.loads(parent_conn.recv_bytes())
                            if pt_rate > self._max_pt_rate:
                                self._max_pt_rate = pt_rate
                            cur_seed_vector = np.concatenate((cur_seed_line_vector, cur_seed_pt_vector))
            
                        else:
                            cur_seed_nac_rate = pickle.loads(parent_conn.recv_bytes())
                            self._cur_nac_rate = cur_seed_nac_rate
                            cur_seed_nac_vector = pickle.loads(parent_conn.recv_bytes())
                            cur_pnac_rate = pickle.loads(parent_conn.recv_bytes())
                            pnac_map = pickle.loads(parent_conn.recv_bytes())
                            
                            
                            PNAC.global_pnac_map = pnac_map

                            is_predict_true = pickle.loads(parent_conn.recv_bytes())
                            # answers = pickle.loads(parent_conn.recv_bytes())
                            answers_ob_tl = pickle.loads(parent_conn.recv_bytes())
                            answers_planning = pickle.loads(parent_conn.recv_bytes())
                            answers_control = pickle.loads(parent_conn.recv_bytes())
                            answers = [
                                1 if answers_ob_tl[i] == 1 and answers_planning[i] == 1 and answers_control[i] == 1 else 0
                                for i in range(len(answers_ob_tl))
                            ]
                            ious = pickle.loads(parent_conn.recv_bytes())
                            tdists = pickle.loads(parent_conn.recv_bytes())
                            steer_diff = pickle.loads(parent_conn.recv_bytes())
                            self.iou.append(ious)
                            self.tdist.append(tdists)
                            self.steer_diff.append(steer_diff)


                            if len(self._total_nac_vector) == 0:
                                self._total_nac_vector = cur_seed_nac_vector.copy()
                            else:
                                self._total_nac_vector = [a | b for a, b in zip(self._total_nac_vector, cur_seed_nac_vector)]
                            print(f'len(cur_seed_nac_vector), len(self._total_nac_vector):{len(cur_seed_nac_vector)}, {len(self._total_nac_vector)}')
                            self._max_nac_rate = sum(self._total_nac_vector) / float(len(self._total_nac_vector)) + self._max_pnac*0.1
                            self._total_nac_rate = sum(self._total_nac_vector) / float(len(self._total_nac_vector))
                            cur_seed_vector = np.concatenate((cur_seed_line_vector, cur_seed_nac_vector))
                            self._cur_pnac = cur_pnac_rate
                            # self._max_pnac = max(self._max_pnac, cur_pnac_rate)
                        if is_init_seed:
                            self._init_seed_vectors.append(cur_seed_vector.tolist())
                        else:
                            self._mutation_seed_vectors.append(cur_seed_vector.tolist())
                except ValueError as e:
                    print(e)
                    
                    self._error_number += 1
                    


                  
                    # img_array = np.array([np.array(img.resize((1920, 1080))) for img in buf.get_params()['imgs']])
                    # model_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/traffic_light_detection/frozen_graph.pb'
                    # x_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/obstacle_detection/x.npy'
                    # vec_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/obstacle_detection/vec.json'
                    # np.save(x_path ,img_array)
                    # subprocess.run(['python', '/media/lzq/D/lzq/pylot_test/pythonfuzz/pythonfuzz/Coverages/NacCoverage.py', '--model_path', model_path, '--data_path', x_path, '--vec_path', vec_path])
                   

                    # seed_ind = self._corpus.get_corpus_size()-self.batch_size+self._total_executions
                    name = f"{self._total_executions}_{self._total_seedpool_size}_{buf.from_seed_que[-1]}"
                    SaveUtil.saveSeedToPickle(buf, name, config.CRASH_SEED_PATH)
                    # if self.batch_size == 1:
                    #     self.write_sample(pickle.dumps(buf))
                    #     SaveUtil.save_error_seed(f'{self._total_executions}', CRASH_SEED_PATH, buf)
                    #     if not is_init_seed:
                    #         self._corpus.put(buf)
                    # else:
                    #     for i, b in enumerate(bufs):
                    #         for param_name, param_type in param_types.items():
                    #             if not param_type==list:
                    #                 b.get_params()[param_name] = buf.get_params()[param_name]
                    #         self.write_sample(pickle.dumps(b))  
                    #         SaveUtil.save_error_seed(f'{self._total_executions}_{i}', CRASH_SEED_PATH, b)
                    #         if not is_init_seed:
                    #             self._corpus.put(b)
                    exit_code = 76

                    x_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/obstacle_detection/x.npy'
                    img_array = np.array([np.array(img.resize((1920, 1080))) for img in buf.get_params()['imgs']])
                    np.save(x_path ,img_array)
                    
                    pnac = PNAC.PNAC()
                    pnac_map, pnac_rate, cur_seed_nac_vec, cur_seed_nac_rate,total_line_rate, total_1_count, total_1_and_0_count, cur_seed_line_vector, cur_seed_line_rate, batch_nac_vectors =pnac.get_PNAC(global_bacth_size)
                    print('报错情况下的pnac_rate:', pnac_rate)
                    self._cur_pnac = pnac_rate
                    self.batch_nac_vectors_list.append(batch_nac_vectors)
                    #计算行覆盖向量
                    coverage_files = glob.glob(f'{COV_FILE_PATH}/.coverage*')
                    cov = cv.Coverage(branch=True)
                    cov.start()
                    # cov.exclude('@coverage_decorator')
                    # cov.exclude('def coverage_decorator')
                    cov.stop()
                    cov.save()
                    # 逐个合并文件，跳过不兼容的文件
                    for file_path in coverage_files:
                        try:
                            cov.combine(data_paths=[file_path], keep=True)
                        except Exception as e:
                            logging.warning(f"Skipping incompatible coverage file: {file_path}, error: {e}")
                            continue
                    if not os.path.exists(COV_HTML_PATH+"/"+str(self._total_executions)):
                        os.makedirs(COV_HTML_PATH+"/"+str(self._total_executions))
                    if not os.path.exists(COV_REPORT_PATH+"/"+str(self._total_executions)):
                        os.makedirs(COV_REPORT_PATH+"/"+str(self._total_executions))
                
                    cov.html_report(directory=COV_HTML_PATH+"/"+str(self._total_executions))
                    print('报错情况保存覆盖信息')
                    # 命令行模式展示结果
                    with open(COV_REPORT_PATH+"/"+str(self._total_executions)+"/coverage_report_"+str(self._total_executions)+".txt", "w") as file:
                        cov.report(file=file)
                    lineCoverage = LineCoverage()
                    total_line_rate, total_1_count, total_1_and_0_count, cur_seed_line_vector, line_status_vector_map,cur_line_status_vector_map, global_keys =\
                          lineCoverage.get_line_coverage(COV_HTML_PATH+"/"+str(self._total_executions))
                    
                    # 提取错误情况下的分支覆盖信息
                    import io
                    import re
                    total_branch_covered = 0
                    total_branches = 0
                    total_branch_rate = 0.0
                    
                    try:
                        # 使用 report() 方法获取摘要信息
                        report_buffer = io.StringIO()
                        cov.report(file=report_buffer)
                        report_text = report_buffer.getvalue()
                        
                        # 从报告的最后一行（TOTAL行）解析分支信息
                        lines = report_text.strip().split('\n')
                        for line in reversed(lines):
                            if line.startswith('TOTAL') or 'TOTAL' in line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    try:
                                        total_branches = int(parts[3])  # Branch 列
                                        br_part = int(parts[4])  # BrPart 列
                                        # total_branch_covered = total_branches - br_part
                                        total_branch_covered = 0
                                        if total_branches > 0:
                                            total_branch_rate = total_branch_covered / total_branches
                                        break
                                    except (ValueError, IndexError) as e:
                                        logging.debug(f"Could not parse TOTAL line: {e}")
                                        break
                    except Exception as e:
                        logging.warning(f"Could not extract branch coverage from data: {e}")
                        total_branch_rate = 0.0
                    
                    print('报错情况下的行覆盖：', total_line_rate)
                    print('报错情况下的分支覆盖：', total_branch_rate)
                    
                    # 更新分支覆盖统计
                    self._cur_branch_rate = total_branch_rate
                    if total_branch_rate > self._max_branch_rate:
                        self._max_branch_rate = total_branch_rate
                    
                    # self._max_line_rate = max(self._max_line_rate, total_line_rate)
                    if len(self._total_nac_vector) == 0:
                        self._total_nac_vector = cur_seed_nac_vec.copy()
                    else:
                        self._total_nac_vector = [a | b for a, b in zip(self._total_nac_vector, cur_seed_nac_vec)]
                    self._cur_nac_rate = sum(cur_seed_nac_vec)/float(len(cur_seed_nac_vec))
                    self._total_nac_rate = sum(self._total_nac_vector) / float(len(self._total_nac_vector))
                    self._max_nac_rate = sum(self._total_nac_vector) / float(len(self._total_nac_vector)) + self._max_pnac*0.1
                    if self._use_functions == 'pythonfuzz':
                        self._max_all_nac_rate = max(self._max_all_nac_rate, self._cur_nac_rate)
                        self._max_nac_rate = self._max_all_nac_rate
                    elif self._use_functions == 'deephunter':
                        self._max_nac_rate = self._total_nac_rate
                    print(f'len(cur_seed_nac_vector), len(self._total_nac_vector):{len(cur_seed_nac_vec)}, {len(self._total_nac_vector)}')
                    
                    if total_line_rate > self._max_line_rate:
                        self._total_seedpool_size += 1
                        seed_ind = self._total_seedpool_size
                        SaveUtil.saveSeedToPickle(buf, str(seed_ind), config.LOCAL_SEED_POOL)
            
                    if total_line_rate > self._max_line_rate:
                        self._max_line_rate = total_line_rate
                        self._max_1_count = total_1_count
                        self._max_1_and_0_count = total_1_and_0_count
                    

                    #更新混合覆盖率
                    self._max_combinedCoverage = max(self._max_combinedCoverage, (self._max_line_rate +self._max_nac_rate)/2)
                    self._max_pnac = max(self._max_pnac, pnac_rate)
                    # self._pnac_map = pnac_map

                    #记录触发错误的种子的覆盖向量，以分析覆盖向量与错误类型之间的关系
                    os.makedirs(config.ERROR_SEEDS_VECTOR, exist_ok=True)
   
                    v_map = {'line_vector':cur_seed_line_vector, 'nac_vector':cur_seed_nac_vec}
                    with open(f'{config.ERROR_SEEDS_VECTOR}/{self._total_executions + 1}_error.pickle', "wb") as file:
                        pickle.dump(v_map, file)
                    self.seed_parents.append(buf.from_seed_que[-1])
                    self.seed_vectors.append(v_map)
                    self.plt_line_cov.append(self._max_line_rate)
                    self.plt_nac_cov.append(self._max_nac_rate)
                    self.plt_combine_cov.append(self._max_combinedCoverage)
                    self.plt_pnac_cov.append(self._max_pnac)
                    self.pnac_maps.append(pnac_map)
                    self.line_status_vector_maps.append(line_status_vector_map)
                    self.cur_line_status_vector_maps.append(cur_line_status_vector_map)
                    
                    self.code_errors.append(self._error_number)
                    self.model_errors.append(self._predict_error_number)
                    self.control_errors.append(self._control_error_number)
                    self.planning_errors.append(self._planning_error_number)
                    self.iou.append([])
                    self.tdist.append([])
                    self.steer_diff.append([])
                    self.cur_seed_errors.append({'error_number':1, 'predict_error_number':0, 
                                                 'planning_error_number':0, 'control_error_number':0})
                    self._total_executions += 1
       
                    if (self._total_executions == 98 or self._total_executions >= 99 and self._total_executions <= 100 or self._total_executions > 100 and self._total_executions % 25 == 0):
                        plt.draw_coverage(self.plt_line_cov, self.plt_nac_cov, self.plt_combine_cov)
                        plt.draw_error(self.code_errors, self.model_errors)
                        m = {'plt_line_cov':self.plt_line_cov, 'plt_nac_cov':self.plt_nac_cov, 'plt_pnac_cov':self.plt_pnac_cov,
                            'plt_combine_cov':self.plt_combine_cov, 'code_errors':self.code_errors,
                            'model_errors': self.model_errors, 'planning_errors': self.planning_errors,
                            'control_errors': self.control_errors, 'ious': self.iou, 'tdist': self.tdist, 
                            'steer_diff': self.steer_diff, 'seed_vectors': self.seed_vectors, 
                            'pnac_map': self.pnac_maps,'line_status_vector_maps':self.line_status_vector_maps,
                            'cur_line_status_vector_maps':self.cur_line_status_vector_maps,
                            'seed_parents': self.seed_parents, 'cur_seed_errors': self.cur_seed_errors,
                            'batch_nac_vectors': self.batch_nac_vectors_list}
                        with open(f'/home/lzq/result/datas/{self._total_executions}.json', "w", encoding="utf-8") as f:
                            json.dump(m, f, ensure_ascii=False, indent=4)
                    
                    # 在iterations=100时自动保存总的覆盖率报告
                    if self._total_executions == 30:
                        save_total_coverage_reportM()
                    if self._total_executions == 100:
                        save_total_coverage_report()
                    
                    
                    
                    

                    self.log_stats("NEW", buf)
                    continue
                #更新总执行次数 self._total_executions 和当前采样窗口内的执行次数 self._executions_in_sample。
                self._total_executions += 1
                self._executions_in_sample += 1
                rss = 0
                '''
                如果新覆盖率 total_coverage 超过当前覆盖率 self._line_coverage，记录 "NEW" 日志，更新总覆盖率，并将新的输入添加到语料库。
                如果一次生成中，生成的测试用例所达覆盖率最大，则当前采样窗口结束
                '''
                # print("line_rate:", total_line_rate)
                
                    
                


                print("is_init_seed", is_init_seed)
                increase_cov = False

                self._max_all_nac_rate = max(self._max_all_nac_rate, self._cur_nac_rate)

                if self._use_functions == 'pythonfuzz':
                    self._max_nac_rate = self._max_all_nac_rate
                    if total_line_rate > self._max_line_rate:
                        increase_cov = True
                        self._total_seedpool_size += 1
                        seed_ind = self._total_seedpool_size
                        SaveUtil.saveSeedToPickle(buf, str(seed_ind), config.LOCAL_SEED_POOL)
                elif self._use_functions == 'deephunter':
                    self._max_nac_rate = self._total_nac_rate
                    if self._max_nac_rate > self.pre_max_nac_rate: 
                        increase_cov = True
                        self._total_seedpool_size += 1
                        seed_ind = self._total_seedpool_size
                        SaveUtil.saveSeedToPickle(buf, str(seed_ind), config.LOCAL_SEED_POOL)
                elif self._use_functions == 'codelfuzz1' or (self._use_functions == 'neude' and self._neude_method == 'neude_gw'):
                    if total_line_rate > self._max_line_rate or self._max_nac_rate > self.pre_max_nac_rate or total_1_count > self._cur_1_count : 
                        increase_cov = True 
                        if not is_init_seed: 
                            self._total_seedpool_size += 1 
                            seed_ind = self._total_seedpool_size 
                            SaveUtil.saveSeedToPickle(buf, str(seed_ind), config.LOCAL_SEED_POOL) 
                elif self._use_functions == 'codelfuzz2' or (self._use_functions == 'neude' and self._neude_method == 'neude_pn'):
                    if cur_pnac_rate > self._max_pnac or (self._pnac_map != None and len(pnac_map) > len(self._pnac_map)) : 
                        increase_cov = True 
                        # print(total_coverage) 
                        # 将buf添加到语料库中 
                        if not is_init_seed: 
                            self._total_seedpool_size += 1 
                            seed_ind = self._total_seedpool_size 
                            SaveUtil.saveSeedToPickle(buf, str(seed_ind), config.LOCAL_SEED_POOL) 

                self._cur_line_rate = total_line_rate
                self._cur_1_count = total_1_count
                self._cur_1_and_0_count = total_1_and_0_count

                if total_line_rate > self._max_line_rate:
                    self._max_line_rate = total_line_rate
                    self._max_1_count = total_1_count
                    self._max_1_and_0_count = total_1_and_0_count
                 

                self._pnac_map = pnac_map
                self._max_pnac = max(self._max_pnac, cur_pnac_rate)
                if global_has_model and global_nc_cov:
                    print('answers_planning:',answers_planning)
                    print('answers_control:',answers_control)
                    self._predict_error_number += answers_ob_tl.count(0) 
                    self._planning_error_number += answers_planning.count(0)
                    self._control_error_number += answers_control.count(0)
                    name = f"{self._total_executions}_{self._total_seedpool_size}_{buf.from_seed_que[-1]}"
                    if answers_ob_tl.count(0) > 0:
                        path=SaveUtil.saveSeedToPickle(buf, name, config.PREDICT_ERROR_SEED_PATH)
                        print(f'Perception Error, the seed has saved to {path}')
                    if answers_planning.count(0) > 0:
                        path=SaveUtil.saveSeedToPickle(buf, name, config.PLANNING_ERROR_SEED_PATH)
                        print(f'Planning Error, the seed has saved to {path}')
                    if answers_control.count(0) > 0:
                        path=SaveUtil.saveSeedToPickle(buf, name, config.CONTROL_ERROR_SEED_PATH)
                        print(f'Control Error, the seed has saved to {path}')
                # if global_has_model and not is_predict_true:
                #     self._predict_error_number += answers.count(0)
                #     if self.batch_size == 1:
                #         SaveUtil.save_error_seed(f'{self._total_executions}_{i}', PREDICT_ERROR_SEED_PATH, buf)
                #     else:
                #         for i, seed in enumerate(bufs):
                #             if answers[i]==0:
                #                 SaveUtil.save_error_seed(f'{self._total_executions}_{i}', PREDICT_ERROR_SEED_PATH, seed)
                #     if not is_init_seed and not increase_cov:
                #         if self.batch_size == 1:
                #             self._corpus.put(buf)
                #         else:
                #             for i, seed in enumerate(bufs):
                #                 if answers[i]==0:
                #                     self._corpus.put(seed)
                            # rss = self.log_stats("NEW")
                            #保存种子图片到本地
                
                # SaveUtil.save(buf, self._total_executions, DATA_SAVE_PATH)
                # else:

                #     '''
                #         否则，如果自上次采样以来时间超过了采样窗口 SAMPLING_WINDOW，记录 "PULSE" 日志。
                #     '''
                #     if (time.time() - self._last_sample_time) > SAMPLING_WINDOW:
                #         rss = self.log_stats('PULSE')
                if not global_nc_cov:
                    self._combinedCoverage = (self._cur_1_count + self._pre_pt_rate * self._pre_region_length) / (self._cur_1_and_0_count + self._pre_region_length)
                else:
                    # self._combinedCoverage = (self._cur_1_count + sum(self._total_nac_vector)) / (self._cur_1_and_0_count + len(self._total_nac_vector))
                    self._combinedCoverage = (self._max_line_rate + self._max_nac_rate) / 2
                if self._combinedCoverage > self._max_combinedCoverage:
                    self._max_combinedCoverage = self._combinedCoverage
                
                # print("init_seed_vectors.size:", len(self._init_seed_vectors))
                # print("mutation_seed_vectors.size:", len(self._mutation_seed_vectors))

                # if self._total_executions % self._cov_vec_save_iter == 0:
                #     i = self._total_executions / self._cov_vec_save_iter
                #     np.save(f"cov_vec_{i}.npy", np.array(self._init_seed_vectors[(i-1)*self._cov_vec_save_iter:]))

                os.makedirs(config.ERROR_SEEDS_VECTOR, exist_ok=True)
                v_map = {'line_vector':cur_seed_line_vector,'nac_vector':cur_seed_nac_vector}
                with open(f'{config.ERROR_SEEDS_VECTOR}/{self._total_executions}_normal.pickle', "wb") as file:
                    pickle.dump(v_map, file)
                self.seed_parents.append(buf.from_seed_que[-1])
                self.seed_vectors.append(v_map)
                self.plt_line_cov.append(self._max_line_rate)
                self.plt_nac_cov.append(self._max_nac_rate)
                self.plt_combine_cov.append(self._max_combinedCoverage)
                self.plt_pnac_cov.append(self._max_pnac)
                self.pnac_maps.append(self._pnac_map)
                self.code_errors.append(self._error_number)
                self.model_errors.append(self._predict_error_number)
                self.control_errors.append(self._control_error_number)
                self.planning_errors.append(self._planning_error_number)
                self.cur_seed_errors.append({'error_number':0, 'predict_error_number':answers_ob_tl.count(0) , 
                                                 'planning_error_number':answers_planning.count(0), 
                                                 'control_error_number':answers_control.count(0)})
                
                if (self._total_executions == 98 or self._total_executions >= 99 and self._total_executions <= 100 or self._total_executions > 100 and self._total_executions % 25 == 0):
                    plt.draw_coverage(self.plt_line_cov, self.plt_nac_cov, self.plt_combine_cov)
                    plt.draw_error(self.code_errors, self.model_errors)
                    m = {'plt_line_cov':self.plt_line_cov, 'plt_nac_cov':self.plt_nac_cov, 'plt_pnac_cov':self.plt_pnac_cov,
                            'plt_combine_cov':self.plt_combine_cov, 'code_errors':self.code_errors,
                            'model_errors': self.model_errors, 'planning_errors': self.planning_errors,
                            'control_errors': self.control_errors, 'ious': self.iou, 
                            'tdist': self.tdist, 'steer_diff': self.steer_diff, 'seed_vectors': self.seed_vectors,
                            'pnac_map': self.pnac_maps,'line_status_vector_maps':self.line_status_vector_maps,
                            'cur_line_status_vector_maps':self.cur_line_status_vector_maps,
                            'seed_parents': self.seed_parents, 'cur_seed_errors':self.cur_seed_errors,
                            'batch_nac_vectors': self.batch_nac_vectors_list}
                    with open(f'/home/lzq/result/datas/{self._total_executions}.json', "w", encoding="utf-8") as f:
                        json.dump(m, f, ensure_ascii=False, indent=4)
                
                # 在iterations=100时自动保存总的覆盖率报告
                if self._total_executions == 30:
                    save_total_coverage_reportM()
                if self._total_executions == 100:
                    save_total_coverage_report()
                

                rss = self.log_stats("NEW", buf)


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
        except KeyboardInterrupt as e:
            save_total_coverage_report()




def get_function_line_range(func):
    lines, start_line = inspect.getsourcelines(func)
    end_line = start_line + len(lines) - 1
    file_path = inspect.getsourcefile(func)
    return file_path, start_line, end_line


def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):  # 获取所有子进程
            print(f"Terminating child process {child.pid}")
            child.kill()  # 优雅终止子进程
        parent.kill()  # 终止主进程
    except psutil.NoSuchProcess:
        pass

def get_fuzz_pids():
    """获取所有 fuzz.py 相关进程的 PID 并按升序排序"""
    result = subprocess.run("ps -eo pid,cmd | grep fuzz_list.py | grep -v grep | sort -n",
                            shell=True, capture_output=True, text=True)
    
    # 提取 PID 列表
    pids = [int(line.split()[0]) for line in result.stdout.strip().split("\n") if line]
    
    return pids

def terminate_extra_fuzz_pids():
    """保留最小的两个 PID，强制终止其余 fuzz.py 进程，并确保资源回收"""
    pids = get_fuzz_pids()

    if len(pids) > 2:
        # 让第二个进程成为“收割者”，确保回收资源
        second_pid = pids[1]
        third_pid = pids[2]

        for pid in pids[2:]:  # 终止第 3 个及以后的进程
            print(f"Force killing process PID: {pid}")
            os.kill(pid, signal.SIGKILL)  # 强制终止

        print(f"Process {second_pid} will reap zombies.")

        # 让第二个进程执行 wait() 来回收僵尸进程
        subprocess.run(f"sudo nsenter -t {second_pid} -m -p -- sh -c 'wait {third_pid}'", shell=True)


    else:
        print("Less than 3 fuzz.py processes, no need to terminate.")

'''
保存总的覆盖率报告（HTML和文本报告）
'''
def save_total_coverage_report():
    """合并所有覆盖率文件并生成总的覆盖率报告"""
    coverage_files = glob.glob(f'{ITER_COV_REPORT_PATH}/.coverage*')
    cov = cv.Coverage(branch=True)
    cov.start()
    cov.stop()
    cov.save()
    # 逐个合并文件，跳过不兼容的文件
    for file_path in coverage_files:
        try:
            cov.combine(data_paths=[file_path], keep=True)
        except Exception as e:
            logging.warning(f"Skipping incompatible coverage file: {file_path}, error: {e}")
            continue
    os.makedirs(COV_HTML_PATH+"/total100", exist_ok=True)
    os.makedirs(COV_REPORT_PATH+"/total100", exist_ok=True)

    cov.html_report(directory=COV_HTML_PATH+"/total100")
    # 命令行模式展示结果
    with open(COV_REPORT_PATH+"/total100"+"/coverage_report_total100"+".txt", "w") as file:
        cov.report(file=file)
    logging.info("Total100 coverage report saved successfully.")

def save_total_coverage_reportM():
    """合并所有覆盖率文件并生成总的覆盖率报告"""
    coverage_files = glob.glob(f'{ITER_COV_REPORT_PATH}/.coverage*')
    cov = cv.Coverage(branch=True)
    cov.start()
    cov.stop()
    cov.save()
    # 逐个合并文件，跳过不兼容的文件
    for file_path in coverage_files:
        try:
            cov.combine(data_paths=[file_path], keep=True)
        except Exception as e:
            logging.warning(f"Skipping incompatible coverage file: {file_path}, error: {e}")
            continue
    os.makedirs(COV_HTML_PATH+"/totalM", exist_ok=True)
    os.makedirs(COV_REPORT_PATH+"/totalM", exist_ok=True)

    cov.html_report(directory=COV_HTML_PATH+"/totalM")
    # 命令行模式展示结果
    with open(COV_REPORT_PATH+"/totalM"+"/coverage_report_totalM"+".txt", "w") as file:
        cov.report(file=file)
    logging.info("TotalM coverage report saved successfully.")

'''
在每轮测试中清空相应的结果保存文件夹
'''
def recreate_folder():

    if os.path.exists(OB_TL_PREDICTIONS_NPYS):
        shutil.rmtree(OB_TL_PREDICTIONS_NPYS)
    os.makedirs(OB_TL_PREDICTIONS_NPYS, exist_ok=True)

    if os.path.exists(CONTROL_PREDICTIONS_NPYS):
        shutil.rmtree(CONTROL_PREDICTIONS_NPYS)
    os.makedirs(CONTROL_PREDICTIONS_NPYS, exist_ok=True)

    if os.path.exists(PLANNING_PREDICTIONS_NPYS):
        shutil.rmtree(PLANNING_PREDICTIONS_NPYS)
    os.makedirs(PLANNING_PREDICTIONS_NPYS, exist_ok=True)

    if os.path.exists(ITER_COV_REPORT_PATH):
        # 删除上一轮的覆盖信息文件
        shutil.rmtree(ITER_COV_REPORT_PATH)
    os.makedirs(ITER_COV_REPORT_PATH, exist_ok=True)

    if os.path.exists(CRASH_SEED_PATH):
        shutil.rmtree(CRASH_SEED_PATH)
    os.makedirs(CRASH_SEED_PATH, exist_ok=True)

    if os.path.exists(config.PREDICT_ERROR_SEED_PATH):
        shutil.rmtree(config.PREDICT_ERROR_SEED_PATH)
    os.makedirs(config.PREDICT_ERROR_SEED_PATH, exist_ok=True)

    if os.path.exists(config.PLANNING_ERROR_SEED_PATH):
        shutil.rmtree(config.PLANNING_ERROR_SEED_PATH)
    os.makedirs(config.PLANNING_ERROR_SEED_PATH, exist_ok=True)

    if os.path.exists( config.CONTROL_ERROR_SEED_PATH):
        shutil.rmtree(config.CONTROL_ERROR_SEED_PATH)
    os.makedirs(config.CONTROL_ERROR_SEED_PATH, exist_ok=True)

def update_config(file_path, updates):
    # 读取配置文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for param_name, new_value in updates.items():
        found = False  # 标记参数是否在文件中找到
        for i in range(len(lines)):
            if lines[i] is not None and lines[i].startswith(param_name):
                found = True
                if new_value is False:
                    # 如果参数值为 False，删除该参数
                    lines[i] = None  # 标记为删除
                elif new_value is not True:
                    lines[i] = f"{param_name}={new_value}\n"
                break  # 找到后退出内层循环

        # 如果参数未找到且值为 True，则添加该参数
        if new_value is True and not found:
            lines.append(f"{param_name}\n")  # 添加参数行
        elif not isinstance(new_value, bool) and not found:
            lines.append(f"{param_name}={new_value}\n")

    # 过滤掉标记为删除的行
    lines = [line for line in lines if line is not None]

    # 将更新后的内容写回文件
    with open(file_path, 'w') as file:
        file.writelines(lines)

