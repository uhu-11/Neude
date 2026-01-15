
model_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/traffic_light_detection/faster-rcnn/saved_model.pb'
h5_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/traffic_light_detection/faster-rcnn/model.h5'
import tensorflow as tf
import os
print(f"文件存在: {os.path.exists(model_path)}")
# model = tf.keras.models.load_model(model_path)
def load_frozen_graph(pb_path):
    try:
        # 尝试加载为冻结图
        graph = tf.Graph()
        with tf.io.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with graph.as_default():
            tf.import_graph_def(graph_def, name="")

        return graph
    except Exception as e:
        print(f"加载冻结图时出错: {e}")

def convert_pb_to_h5(pb_path, h5_path):
    graph_def = load_frozen_graph(pb_path)
    print(f"图中的操作数量: {len(graph_def.get_operations())}")
    # with tf.compat.v1.Session(graph=graph_def) as graph:
    #     # tf.import_graph_def(graph_def, name="")
        
    #     # 打印图中的操作以进行调试
    #     operations = graph.get_operations()
    #     print(f"图中的操作数量: {len(operations)}")
    #     for op in operations:
    #         print(op.name)

    #     # if len(operations) == 0:
    #     #     raise ValueError("图中没有操作，请检查输入的 .pb 文件。")

    #     # 创建 Keras 模型
    #     model = tf.keras.Model(inputs=operations[0].outputs[0], 
    #                            outputs=operations[-1].outputs[0])
    #     model.save(h5_path)

# 检查文件
convert_pb_to_h5(model_path, h5_path)







#测试是否为SavedModel形式：
# # 检查 SavedModel 是否tf.saved_model.load存在
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"指定的 SavedModel 路径不存在: {model_path}")

# # 加载 SavedModel
# model = tf.keras.models.load_model(model_path)

# # 保存为 .h5 格式
# model.save(h5_path)
# print(f"模型已成功保存为 {h5_path}")