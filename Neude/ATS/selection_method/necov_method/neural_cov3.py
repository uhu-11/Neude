import time
import numpy as np
# from keras.engine.saving import load_model
import sys
sys.path.append('/media/lzq/D/lzq/pylot_test/pythonfuzz/ATS')
from selection_method.necov_method import metrics
import tensorflow as tf


##############
# cal coverage

    
def load_frozen_graph_function(pb_file, input_name, output_names):
    # 读取冻结图
    
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        
    # 定义一个函数来导入图
    def _imports():
        tf.compat.v1.import_graph_def(graph_def, name="")  # 使用空前缀
    wrapped_import = tf.compat.v1.wrap_function(_imports, [])
    inputs = wrapped_import.graph.get_tensor_by_name(input_name)
    for output_name in output_names:
        # 根据名称获取输入和输出张量
        outputs = wrapped_import.graph.get_tensor_by_name(output_name)
        func = wrapped_import.prune(inputs, outputs)
        layers = [('conv', outputs, func)]
        
    return inputs, layers
        # 返回一个 ConcreteFunction，其输入输出分别为上面获得的张量
        # return [('conv', wrapped_import.prune(inputs, outputs))]

class CovInit(object):
    def __init__(self, model_path):
        self.model_path = model_path
        input_names = "image_tensor:0"
        output_names = ["FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block1/unit_1/bottleneck_v1/shortcut/Conv2D:0"]
        with tf.Graph().as_default() as graph_1:
            self.input_layer, self.layers = load_frozen_graph_function(self.model_path, input_names, output_names)
        
    def get_layers(self):
        return self.layers
    def get_input_layer(self):
        return self.input_layer
    

class CovRank(object):
    def __init__(self, cov_initer: CovInit, model_path, x_s):
        self.cov_initer = cov_initer
        self.model_path = model_path
        self.x_s = x_s

    def get_layers(self):
        return self.cov_initer.layers

    # def load_ori_model(self):
    #     return load_model(self.model_path)



    def cal_nac_cov(self, t=0.75, only_ctm=False):
        layers = self.get_layers()
        nac = metrics.nac(self.x_s, layers, t=t)

        if only_ctm:
            rank_lst2 = nac.rank_2(self.x_s)
            return None, None, None, None, rank_lst2, None
        else:
            rate, vector = nac.fit()
            # rank_lst = nac.rank_fast(self.x_s)
            # rank_lst2 = nac.rank_2(self.x_s)
            return rate, vector


if __name__ == '__main__':
    model = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/traffic_light_detection/frozen_graph.pb'
    covinit = CovInit(model)
    dummy_input = np.random.randint(0, 255, (1, 640, 640, 3), dtype=np.uint8)

    dummy_input_tensor = tf.convert_to_tensor(dummy_input, dtype=tf.uint8)
    
    # 调用函数获取中间节点输出
    # result = covinit.layers[0][2](dummy_input_tensor).numpy().reshape(1, -1, covinit.layers[0][1].shape[-1])
    # print(result.shape)
    # print(type(result))
    # temp = np.mean(result, axis=1)
    # temp = 1 / (1 + np.exp(-temp))
    # print(temp)
    covrank=CovRank(covinit, model, dummy_input)
    rate, vector = covrank.cal_nac_cov()
    print(rate, vector)