import tensorflow as tf
import argparse

# 兼容 TF2.5
tf.compat.v1.disable_eager_execution()  # 禁用 Eager Execution，兼容 TF1 代码
tf.compat.v1.reset_default_graph()  # 重置计算图

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def to_graph():
    # 加载 SavedModel
    model = tf.saved_model.load('/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/obstacle_detection/faster-rcnn')

    # 获取计算图
    concrete_func = model.signatures["serving_default"]
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_graph_def = frozen_func.graph.as_graph_def()

    # 保存为新的 Frozen Graph
    with tf.io.gfile.GFile('/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/obstacle_detection/frozen_graph.pb', "wb") as f:
        f.write(frozen_graph_def.SerializeToString())

import numpy as np
def network_structure(args):
    args.model = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/traffic_light_detection/frozen_graph.pb'
    model_path = args.model
    to_graph()
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()
        output_graph_def = tf.compat.v1.GraphDef()

        # 读取 .pb 文件
        with open(model_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        graph = tf.compat.v1.get_default_graph()
        # for node in output_graph_def.node:
        #   print(node.name)


        # 获取中间层张量：例如获取 RPN 卷积层的输出
        tensor_name = 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block1/unit_1/bottleneck_v1/shortcut/Conv2D:0'
        intermediate_tensor = graph.get_tensor_by_name(tensor_name)
        
        # 构造输入数据（这里需要根据模型实际输入构造数据）
        # 此处示例构造一个随机的 uint8 图像，假设输入尺寸为640x640
        dummy_input = np.random.randint(0, 255, (1, 640, 640, 3), dtype=np.uint8)
        
        # 获取输入节点（假设输入节点名称为 image_tensor:0）
        input_tensor = sess.graph.get_tensor_by_name("image_tensor:0")
        
        # 运行会话，获取中间节点输出
        intermediate_output = sess.run(intermediate_tensor, feed_dict={input_tensor: dummy_input})
        print("节点 {} 的输出形状: {}".format(tensor_name, intermediate_output.shape))






        # # 打印模型节点信息
        # print("%d ops in the graph." % len(output_graph_def.node))
        # op_names = [tensor.name for tensor in output_graph_def.node]
        # print(op_names)
        # print('=======================================================')

        # # 记录日志，供 TensorBoard 可视化
        # log_dir = 'log_graph_' + args.model
        # summary_writer = tf.compat.v1.summary.FileWriter(log_dir, graph)
        # print(f"TensorBoard logs saved in: {log_dir}")

        # # 打印张量信息
        # cnt = 0
        # print("%d tensors in the graph." % len(graph.get_operations()))
        # for tensor in graph.get_operations():
        #     print(tensor.name, tensor.values())
        #     cnt += 1
        #     if args.n and cnt == args.n:
        #         break

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, help="model name to look")
    # parser.add_argument('--n', type=int, help="the number of first several tensor names to look")
    # args = parser.parse_args()
    # i=0
    # while i<3:
    #     model = tf.saved_model.load('/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/traffic_light_detection/faster-rcnn')

    #     network_structure(args)
    #     i+=1
    to_graph()




