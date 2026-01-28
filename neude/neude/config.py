import os

# 获取项目根目录（pylot_Neude）
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# pylot文件夹路径
_PYLOT_DIR = os.path.join(_PROJECT_ROOT, 'pylot')

IMAGE_SAVE_PATH = 'images'
GENERATED_IMAGES_PATH = 'generated_images'
COV_HTML_PATH = "covhtml"
COV_REPORT_PATH = "covreport"
SEED_SAVE_PATH = 'seeds'
#记录代码、模型、规划、控制错误
CRASH_SEED_PATH = 'error_seeds/crash_seeds'
PREDICT_ERROR_SEED_PATH = 'error_seeds/predict_error_seeds'
PLANNING_ERROR_SEED_PATH = 'error_seeds/planning_error_seeds'
CONTROL_ERROR_SEED_PATH = 'error_seeds/control_error_seeds'


DATA_SAVE_PATH = 'datas'
ITER_COV_REPORT_PATH = 'coverageFiles'
PROJECT_ERROR_LOG = os.path.join(_PROJECT_ROOT, 'error_log.txt')

#保存pylot运行结果的文件夹
OB_TL_PREDICTIONS_NPYS = os.path.join(_PYLOT_DIR, 'predictions', 'obstacle_trafficlight')
CONTROL_PREDICTIONS_NPYS = os.path.join(_PYLOT_DIR, 'predictions', 'control')
PLANNING_PREDICTIONS_NPYS = os.path.join(_PYLOT_DIR, 'predictions', 'planning')
#记录具体的系统代码报错信息
ERROR_INFOS_DIR = os.path.join(_PYLOT_DIR, 'error_infos')
ERROR_SEEDS_VECTOR = os.path.join(_PYLOT_DIR, 'error_seeds_vectors')

'''
加入到种子池中的新种子所保存的位置，里面保存的种子是batch_size个种子合并之后的种子，
想要获取batch_size个种子的话，直接读取相应的文件就行。
'''
LOCAL_SEED_POOL = os.path.join(_PYLOT_DIR, 'local_seeds_pool')


#记录生成的种子是初始种子池中哪个变异来的
FROM_WITCH_SEED = 'from_seed.json'

EXCLUDE_FILE=['pid', 'coverage_decorator', 'lane', 'dependencies']

COV_FILE_PATH = os.path.join(_PROJECT_ROOT, 'coverage_test')

USE_FUNCTIONS = 'codelfuzz2'