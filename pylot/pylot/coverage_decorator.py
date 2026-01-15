import coverage
import os
import signal
import multiprocessing as mp
from functools import wraps
from pylot.global_var import COV_FILE_PATH
import time
import traceback
import pickle
import numpy as np
def coverage_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # 开始覆盖率记录

            try:
                i = np.load('/media/lzq/D/lzq/pylot_test/img_ind.npy', allow_pickle=True)
                i = int(i)
            except Exception as e:
                print('读取i失败了')
                i=9
            
            cov = coverage.coverage(data_file=f'{COV_FILE_PATH}/.coverage.{i}.{mp.current_process().pid}.{func.__name__}.{time.time()}', branch=True)
            cov.start()
   
            
            # 调用原始方法
            result = func(*args, **kwargs)
            # cov.stop()
            # 保存覆盖率数据
            cov.save()
            return result
        except Exception as e:
            # cov.stop()
            cov.save()
            # 处理异常
            error_message = traceback.format_exc()
            with open('/media/lzq/D/lzq/pylot_test/error_log.txt', 'w') as f:
                f.write("error:" + str(error_message))
            os.kill(os.getppid(), signal.SIGINT)
    return wrapper