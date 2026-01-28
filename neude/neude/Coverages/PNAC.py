import inspect

from neude import tracer
from neude.Coverages.LineCoverage import LineCoverage
from neude.utils import htmlpaser
from neude import config
import glob
import coverage
import numpy as np
import json
import subprocess
import shutil
import os

global_pnac_map={}
class PNAC():
    def __init__(self) -> None:
        self.lineCoverage = LineCoverage()
    def get_PNAC(self, batch):
        cur_iter_nac_vec = None
        cur_iter_line_vector=None
        total_line_rate, total_1_count, total_1_and_0_count = 0,0,0
        # 保存每个图片的nac向量
        batch_nac_vectors = []

        model_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/obstacle_detection/frozen_graph.pb'
        x_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/obstacle_detection/x.npy'
        x_i_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/obstacle_detection/x_i.npy'
        vec_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/obstacle_detection/vec_i.json'
        x = np.load(x_path, allow_pickle=True)
        print('一共的批次：', batch)
        for i in range(batch):
            pattern1 = f'{config.COV_FILE_PATH}/.coverage.{i}*'
            pattern2 = f'{config.COV_FILE_PATH}/.coverage.main*'
            files1, files2 = glob.glob(pattern1), glob.glob(pattern2)
            files = files1 + files2
            # print('files', files)
            if not files:
                continue

            # 创建 Coverage 实例，指定输出合并后的 data_file
            # print(f'读取了第{i}轮的覆盖文件')
            import logging
            
            # 创建合并后的 Coverage 对象（branch=True）
            cov = coverage.Coverage(data_file=f'.coverage.combined.{i}', branch=True)
            
            # 如果 files 不为空，尝试从第一个文件开始合并
            if files:
                # 先尝试合并第一个文件作为基础
                first_file = files[0]
                try:
                    cov.combine(data_paths=[first_file], keep=True)
                    # 如果成功，继续合并其他文件
                    for file_path in files[1:]:
                        try:
                            cov.combine(data_paths=[file_path], keep=True)
                        except Exception as e:
                            error_msg = str(e)
                            if "Can't combine statement coverage data with branch data" in error_msg:
                                logging.warning(f"Skipping incompatible coverage file (no branch data): {file_path}")
                            else:
                                logging.warning(f"Skipping coverage file due to error: {file_path}, error: {error_msg}")
                            continue
                except Exception as e:
                    # 如果第一个文件也不兼容，创建空的 coverage 对象
                    error_msg = str(e)
                    if "Can't combine statement coverage data with branch data" in error_msg:
                        logging.warning(f"First file incompatible (no branch data): {first_file}, creating empty coverage")
                        cov.start()
                        cov.stop()
                        cov.save()
                        # 尝试合并其他文件
                        for file_path in files[1:]:
                            try:
                                cov.combine(data_paths=[file_path], keep=True)
                            except Exception as e2:
                                error_msg2 = str(e2)
                                if "Can't combine statement coverage data with branch data" in error_msg2:
                                    logging.warning(f"Skipping incompatible coverage file (no branch data): {file_path}")
                                else:
                                    logging.warning(f"Skipping coverage file due to error: {file_path}, error: {error_msg2}")
                                continue
                    else:
                        # 其他错误，创建空的 coverage 对象
                        cov.start()
                        cov.stop()
                        cov.save()
            else:
                # 如果没有文件，创建空的 coverage 对象
                cov.start()
                cov.stop()
                cov.save()
            # 生成 HTML 报告到目录 html_report_i
            cov.html_report(directory=f'html_report_{i}')
            total_line_rate, total_1_count, total_1_and_0_count, cur_seed_line_vector,_,cur_line_status_vector_map,_ = self.lineCoverage.get_line_coverage(f'html_report_{i}')
            # print('cur_seed_line_vector', cur_seed_line_vector)
            line_vec_sim = []
            for j in range(len(cur_seed_line_vector)):
                if cur_seed_line_vector[j]==1:
                    line_vec_sim.append(j)

            ### 计算整个迭代轮次中batch_size个图片的行覆盖：cur_iter_line_vec##############33
            if cur_iter_line_vector is None:
                cur_iter_line_vector = cur_seed_line_vector
            else:
                cur_iter_line_vector = [a | b for a, b in zip(cur_iter_line_vector, cur_seed_line_vector)]
            ################################################################
            # print(f'第{i}轮计算的行覆盖向量:{cur_iter_line_vector}')
            np.save(x_i_path, np.expand_dims(x[i], axis=0))
            nac_coverage_path = os.path.join(os.path.dirname(__file__), 'NacCoverage.py')
            subprocess.run(['python', nac_coverage_path, '--model_path', model_path, '--data_path', x_i_path, '--vec_path', vec_path])

            with open(vec_path, 'r') as f:
                cur_seed_nac_vec = json.load(f)
            nac_rate = sum(cur_seed_nac_vec)/float(len(cur_seed_nac_vec))
            
            # 保存每个图片的nac向量
            batch_nac_vectors.append(cur_seed_nac_vec.copy())

            # 计算整个迭代轮次中batch_size个图片的nac：cur_iter_nac_vec#############################
            if cur_iter_nac_vec is None:
                cur_iter_nac_vec = cur_seed_nac_vec
            else:
                cur_iter_nac_vec = [a | b for a, b in zip(cur_iter_nac_vec, cur_seed_nac_vec)]
            ##################################################################################
            for key in cur_line_status_vector_map:
                cur_line_status_vector_map[key] = [x for x in cur_line_status_vector_map[key] if x != -1]
            for key in cur_line_status_vector_map:
                cur_line_status_vector_map[key] = [i for i, val in enumerate(cur_line_status_vector_map[key]) if val == 1]
            k = json.dumps(cur_line_status_vector_map, sort_keys=True)
            if k not in global_pnac_map.keys():
                global_pnac_map[k] = cur_seed_nac_vec
            else:
                global_pnac_map[k]=[a | b for a, b in zip(global_pnac_map[k], cur_seed_nac_vec)]
            shutil.rmtree(f'html_report_{i}')
            # print('global_pnac_map的长度', len(global_pnac_map))
        # print('cur_iter_line_vector',cur_iter_line_vector)
        cur_iter_line_cov = calculate_coverage(cur_iter_line_vector)
        
        one_pnac_rates=[]
        for nv in global_pnac_map.values():
            one_pnac_rates.append(sum(nv)/len(nv))

        pnac_rate = np.mean(one_pnac_rates)
        return global_pnac_map, pnac_rate,cur_iter_nac_vec, sum(cur_iter_nac_vec)/float(len(cur_iter_nac_vec)), total_line_rate, total_1_count, total_1_and_0_count, cur_iter_line_vector, cur_iter_line_cov, batch_nac_vectors
        ##返回 PNAC、 当前轮次batch_size个图片的nac、全部的行覆盖、当前轮次batch_size个图片的行覆盖、每个图片的nac向量列表


            
def calculate_coverage(function_vectors):
    total_1_count = 0  # 统计所有向量中1的总数
    total_1_and_0_count = 0  # 统计所有向量中1和0的总数

    # 遍历函数状态向量map
    for v in function_vectors:
        if v==1:
            total_1_count+=1
            total_1_and_0_count+=1
        elif v==0:
            total_1_and_0_count+=1

    # 计算覆盖率：1的总数 / (1和0的总数)
    if total_1_and_0_count == 0:
        return 0  # 避免除以0的情况
    coverage = total_1_count / total_1_and_0_count
    return coverage

