import coverage
import os
from pylot2 import run_pylot_with_flags
import multiprocessing as mp
def test():
    def subrunner():
        a=1
        # a=1
        # a=1
        # a=1
        # a=1
    print("childtest当前进程：",os.getpid())
    config_path = '/media/lzq/D/lzq/pylot_test/pylot/configs/traffic_light.conf'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    print('test run_pylot_with_flags')
    run_pylot_with_flags(config_path)
 

import time

def r():
    a=1
    b=2
    a=1
    a=1
    a=1

if __name__ == '__main__':
   
    # print("main当前进程：",os.getpid())
    # process = mp.Process(target=test)
    cov = coverage.coverage(data_file='{COV_FILE_PATH}/.coverage.gigity')
    cov.start()

    cov.save()
    # cov.combine()
    # # cov.save()
    # cov.html_report(directory="{COV_FILE_PATH}/coverage_html_report_covtest")
    # with open("{COV_FILE_PATH}/coverage_report33333.txt", "w") as file:
    #     cov.report(file=file)
    
    print('the end of test!!!!!!!!')
    time.sleep(4)