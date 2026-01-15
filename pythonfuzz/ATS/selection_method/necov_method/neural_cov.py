import time
import numpy as np
# from keras.engine.saving import load_model
from selection_method.necov_method import metrics
import tensorflow as tf


##############
# cal coverage
#############
# traffic_model_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/traffic_light_detection/faster-rcnn/saved_model.pb'
# params = {"model_path":traffic_model_path}
#X_train:直接使用fuzzer中seed中图像组成的图片池，Y_train是None
# covinit = CovInit(x, None, params)


class CovInit(object):
    def __init__(self, model_path):
        input_layer, layers = get_layers(model_path)
        self.model_path = model_path
        # self.model_name = model_name
        self.input_layer = input_layer
        self.layers = layers



    def get_input_layer(self):
        return self.input_layer

    def get_layers(self):
        return self.layers

class CovRank(object):
    def __init__(self, cov_initer: CovInit, model_path, x_s):
        self.cov_initer = cov_initer
        self.model_path = model_path
        self.x_s = x_s

    def get_layers(self):
        return self.cov_initer.input_layer, self.cov_initer.layers

    # def load_ori_model(self):
    #     return load_model(self.model_path)



    def cal_nac_cov(self, t=0.75, only_ctm=False):
        input_layer, layers = self.get_layers()
        nac = metrics.nac(self.x_s, input_layer, layers, t=t)

        if only_ctm:
            rank_lst2 = nac.rank_2(self.x_s)
            return None, None, None, None, rank_lst2, None
        else:
            rate = nac.fit()
            # rank_lst = nac.rank_fast(self.x_s)
            # rank_lst2 = nac.rank_2(self.x_s)
            return rate



def get_layers(model_path):
    # tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()
        output_graph_def = tf.compat.v1.GraphDef()

        # 读取 .pb 文件
        with open(model_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        graph = tf.compat.v1.get_default_graph()

        input_layer = graph.get_tensor_by_name('image_tensor:0')
        # 获取中间层张量：例如获取 RPN 卷积层的输出
        tensor_names = ['FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block1/unit_1/bottleneck_v1/shortcut/Conv2D:0',
                        'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block1/unit_1/bottleneck_v1/conv1/Conv2D:0',
                        'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block1/unit_2/bottleneck_v1/conv1/Conv2D:0',
                        'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block3/unit_20/bottleneck_v1/conv2/Conv2D:0',
                        'FirstStageBoxPredictor/ClassPredictor/Conv2D:0',
                        'FirstStageBoxPredictor/BoxEncodingPredictor/Conv2D:0',
                        'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_1/bottleneck_v1/conv1/Conv2D:0',
                        'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_2/bottleneck_v1/conv1/Conv2D:0',
                        'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_3/bottleneck_v1/conv2/Conv2D:0',
                        'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_3/bottleneck_v1/conv3/Conv2D:0']
        layers = []
        for tensor_name in tensor_names:
            intermediate_tensor = graph.get_tensor_by_name(tensor_name)
            layers.append(('conv', intermediate_tensor))
        return input_layer, layers
    
        

# def get_layers(model_name, model):
#     input = model.layers[0].output
#     lst = []
#     for index, layer in enumerate(model.layers):
#         if 'activation' in layer.name:
#             lst.append(index)
#     lst.append(len(model.layers) - 1)
#     if model_name == model_conf.LeNet5:
#         layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output, model.layers[6].output,
#                   model.layers[8].output, model.layers[9].output, model.layers[10].output]
#         layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))
#     elif model_name == model_conf.LeNet1:
#         layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output, model.layers[6].output,
#                   model.layers[8].output, ]
#         layers = list(zip(4 * ['conv'] + 1 * ['dense'], layers))
#     elif model_name == model_conf.resNet20:
#         layers = []
#         for index in lst:
#             layers.append(model.layers[index].output)
#         layers = list(zip(19 * ['conv'] + 1 * ['dense'], layers))
#     elif model_name == model_conf.vgg16 or model_name == model_conf.MyVgg16:  # vgg16
#         layers = []
#         for i in range(1, 19):

#             layers.append(model.layers[i].output)
#         for i in range(20, 23):
#             layers.append(model.layers[i].output)
#         layers = list(zip(18 * ['conv'] + 3 * ['dense'], layers))
#     elif model_name == model_conf.MyLeNet5:
#         layers = [model.layers[2].output, model.layers[3].output,
#                   model.layers[5].output, model.layers[6].output,
#                   model.layers[8].output, model.layers[9].output,
#                   model.layers[11].output, model.layers[12].output,
#                   model.layers[14].output, model.layers[15].output,
#                   model.layers[17].output, model.layers[18].output,
#                   model.layers[20].output, model.layers[21].output, model.layers[22].output, ]
#         layers = list(zip(12 * ['conv'] + 3 * ['dense'], layers))

#     else:
#         raise ValueError("model {} do not have coverage layers config info".format(model_name))
#     return input, layers
