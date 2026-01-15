import numpy as np
import sys
sys.path.append('/media/lzq/D/lzq/pylot_test/pythonfuzz')
sys.path.append('/media/lzq/D/lzq/pylot_test/pythonfuzz/PTtool')
sys.path.append('/media/lzq/D/lzq/pylot_test/pythonfuzz/ATS')
from selection_method.necov_method.neural_cov import CovInit, CovRank
from PTtool.pt.TriangularProjection import TriProCover
import tensorflow as tf
class NacCoverage():
    def __init__(self, tf_model_path, x_test):
        self.tf_model_path = tf_model_path
        self.x_test = x_test
        # self.covinit = covinit
    def get_coverage(self):
        covinit = CovInit(self.tf_model_path)
        covrank = CovRank(covinit, self.tf_model_path, self.x_test)
        return covrank.cal_nac_cov()

    
# if __name__=='__main__':
#     tf_model_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/traffic_light_detection/frozen_graph.pb'
#     model_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/traffic_light_detection/faster-rcnn'
#     i=0

#     model = tf.saved_model.load(model_path)
#     dummy_input = np.random.randint(0, 255, (20, 640, 640, 3), dtype=np.uint8)
#     nc = NacCoverage(tf_model_path=tf_model_path, x_test=dummy_input)
#     score, vec = nc.get_coverage()
#     print('score:',score, 'len(vec):', type(vec))


import argparse
import json
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Command-line script")
    parser.add_argument('--model_path', type=str, help="A parameter", required=True)
    parser.add_argument('--data_path', type=str, help="A parameter", required=True)
    parser.add_argument('--vec_path', type=str, help="A parameter", required=True)

    args = parser.parse_args()
    x = np.load(args.data_path)
    model_path = args.model_path
    vec_path = args.vec_path
    nc = NacCoverage(tf_model_path=model_path, x_test=x)
    cur_seed_nac_score, cur_seed_nac_vec = nc.get_coverage()

    with open(vec_path, 'w') as f:
        json.dump(cur_seed_nac_vec, f)
    









