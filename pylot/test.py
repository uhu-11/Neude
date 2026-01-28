import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pylot'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'neude', 'neude'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'neude', 'PTtool'))

import numpy as np
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pylot2 import run_pylot_with_flags
import random
from neude.main import PythonFuzz
import time
@PythonFuzz
def pylot_test(img):
    time.sleep(5)
    img=img.resize((800,600))
    print(img.size)
    traffic_light_path = '/home/lzq/traffic_light'
    # 检查文件夹是否存在
    if os.path.exists(traffic_light_path):
        # 删除现有文件夹
        shutil.rmtree(traffic_light_path)
    
    # 创建新的文件夹
    os.makedirs(traffic_light_path)
    img.save(os.path.join(traffic_light_path, '0.png'))
    config_path = '/media/lzq/D/lzq/pylot_test/pylot/configs/traffic_light.conf'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    run_pylot_with_flags(config_path)
 

if __name__ == '__main__':
    pylot_test()