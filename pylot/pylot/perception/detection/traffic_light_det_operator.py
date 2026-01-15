"""Implements an operator that detects traffic lights."""
import logging

import erdos
import logging
import numpy as np
import os
import pylot.utils
from pylot.perception.detection.traffic_light import TrafficLight, \
    TrafficLightColor
from pylot.perception.detection.utils import BoundingBox2D
from pylot.perception.messages import TrafficLightsMessage
from tensorflow.keras.models import load_model, save_model
from pylot.coverage_decorator import coverage_decorator
import tensorflow as tf
import signal

import sys
sys.path.append('/media/lzq/D/lzq/pylot_test/pylot')

import multiprocessing as mp
import coverage
from pylot.global_var import COV_FILE_PATH


class TrafficLightDetOperator(erdos.Operator):
    """Detects traffic lights using a TensorFlow model.

    The operator receives frames on a camera stream, and runs a model for each
    frame.

    Args:
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        traffic_lights_stream (:py:class:`erdos.WriteStream`): Stream on which
            the operator sends
            :py:class:`~pylot.perception.messages.TrafficLightsMessage`
            messages.
        flags (absl.flags): Object to be used to access absl flags.
    """
    @coverage_decorator
    def __init__(self, camera_stream: erdos.ReadStream,
                 time_to_decision_stream: erdos.ReadStream,
                 traffic_lights_stream: erdos.WriteStream, flags):


        self._iter=0
        tf.compat.v1.enable_eager_execution()
        # Register a callback on the camera input stream.
        camera_stream.add_callback(self.on_frame, [traffic_lights_stream])
        time_to_decision_stream.add_callback(self.on_time_to_decision_update)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._traffic_lights_stream = traffic_lights_stream
        # Load the model from the model file.
        pylot.utils.set_tf_loglevel(logging.ERROR)


        # Only sets memory growth for flagged GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(
            [physical_devices[self._flags.traffic_light_det_gpu_index]], 'GPU')

        tf.config.experimental.set_memory_growth(
            physical_devices[self._flags.traffic_light_det_gpu_index], True)

        # Load the model from the saved_model format file.
        # print('模型路径：',self._flags.traffic_light_det_model_path)
        self._model = tf.saved_model.load(
            self._flags.traffic_light_det_model_path)
     

        self._labels = {
            1: TrafficLightColor.GREEN,
            2: TrafficLightColor.YELLOW,
            3: TrafficLightColor.RED,
            4: TrafficLightColor.OFF
        }
        # Unique bounding box id. Incremented for each bounding box.
        self._unique_id = 0
        # Serve some junk image to load up the model.
        
        # self.__run_model(np.zeros((108, 192, 3), dtype='uint8'))

    @staticmethod
    @coverage_decorator
    def connect(camera_stream: erdos.ReadStream,
                time_to_decision_stream: erdos.ReadStream):
        """Connects the operator to other streams.

        Args:
            camera_stream (:py:class:`erdos.ReadStream`): The stream on which
                camera frames are received.

        Returns:
            :py:class:`erdos.WriteStream`: Stream on which the operator sends
            :py:class:`~pylot.perception.messages.TrafficLightsMessage`
            messages for traffic lights.
        """
        traffic_lights_stream = erdos.WriteStream()
        return [traffic_lights_stream]
    @coverage_decorator
    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))
        self._traffic_lights_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
    @coverage_decorator
    def on_time_to_decision_update(self, msg: erdos.Message):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))

    @erdos.profile_method()
    @coverage_decorator
    def on_frame(self, msg: erdos.Message,
                 traffic_lights_stream: erdos.WriteStream):

        """Invoked whenever a frame message is received on the stream.

        Args:
            msg: A :py:class:`~pylot.perception.messages.FrameMessage`.
            obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which
                the operator sends
                :py:class:`~pylot.perception.messages.TrafficLightsMessage`
                messages for traffic lights.
        """
        # print("运行cov_run_on_frame")
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        assert msg.frame.encoding == 'BGR', 'Expects BGR frames'
        boxes, scores, labels, res_labels_number = self.__run_model(
            msg.frame.as_rgb_numpy_array())
  
        
        transformed_boxes = []
        for box in boxes:
            transformed_box = [
                box[1] * msg.frame.camera_setup.width,    # x1
                box[0] * msg.frame.camera_setup.height,   # y1
                box[3] * msg.frame.camera_setup.width,    # x2
                box[2] * msg.frame.camera_setup.height    # y2
            ]
            transformed_boxes.append(transformed_box)
        transformed_boxes = [[float(tensor.numpy()) for tensor in sublist] for sublist in transformed_boxes]
        res_map = {'labels':res_labels_number, 'boxes' : transformed_boxes}

        # #保存红绿灯模型输出的逻辑，用的时候解除注释
        # np.save(f'/media/lzq/D/lzq/pylot_test/pylot/predictions/obstacle_trafficlight/prediction_{self._iter:08d}.npy', res_map)
        # self._iter+=1
        # print('红绿灯保存npy完毕.......')
        
        traffic_lights = self.__convert_to_detected_tl(
            boxes, scores, labels, msg.frame.camera_setup.height,
            msg.frame.camera_setup.width)
        

        self._logger.debug('@{}: {} detected traffic lights {}'.format(
            msg.timestamp, self.config.name, traffic_lights))
        
        traffic_lights_stream.send(
            TrafficLightsMessage(msg.timestamp, traffic_lights))
        traffic_lights_stream.send(erdos.WatermarkMessage(msg.timestamp))

        if self._flags.log_traffic_light_detector_output:
            msg.frame.annotate_with_bounding_boxes(msg.timestamp,
                                                   traffic_lights)
            msg.frame.save(msg.timestamp.coordinates[0], self._flags.data_path,
                           'tl-detector-{}'.format(self.config.name))
    @coverage_decorator
    def __run_model(self, image_np):
        # print("traffic_light_det_operator model run............")

        # Expand dimensions since the model expects images to have
        # shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        infer = self._model.signatures['serving_default']

        # logging.warn("层的名称：%s", infer.graph)

        result = infer(tf.convert_to_tensor(value=image_np_expanded))

        result_keys = result.keys()

 
        boxes = result['boxes']
        scores = result['scores']
        classes = result['classes']
        num_detections = result['detections']
        num_detections = int(num_detections[0])
        import random
   
        res_labels = [
            self._labels[int(label)] for label in classes[0][:num_detections]
        ]
        res_labels_number = [int(label) for label in classes[0][:num_detections]]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]
        # logging.warn("res_boxed: %s", res_boxes)
        # logging.warn("res_scores: %s", res_scores)
        # logging.warn("res_labels: %s", res_labels)
        return res_boxes, res_scores, res_labels, res_labels_number


  
        
    @coverage_decorator
    def __convert_to_detected_tl(self, boxes, scores, labels, height, width):
        traffic_lights = []
        for index in range(len(scores)):
            if scores[
                    index] > self._flags.traffic_light_det_min_score_threshold:
                bbox = BoundingBox2D(
                    int(boxes[index][1] * width),  # x_min
                    int(boxes[index][3] * width),  # x_max
                    int(boxes[index][0] * height),  # y_min
                    int(boxes[index][2] * height)  # y_max
                )
                traffic_lights.append(
                    TrafficLight(scores[index],
                                 labels[index],
                                 id=self._unique_id,
                                 bounding_box=bbox))
                self._unique_id += 1
        # print('检测到的红绿灯的长度：', len(traffic_lights))
        return traffic_lights

# def conv_tfmodel_to_keras(infer):

#     # 创建 Keras 输入层，使用 None 表示动态大小
#     inputs = tf.keras.Input(shape=(None, None, 3))  # 输入形状为 (height, width, channels)

#     # 通过模型的推理函数获取输出
#     # 注意：这里 infer 需要接受一个具体的输入张量
#     # 由于 infer 不能直接接受 Keras 输入层，您需要使用 Lambda 层来调用 infer
#     outputs = tf.keras.layers.Lambda(lambda x: infer(tf.convert_to_tensor(x)))(inputs)

#     # 创建 Keras 模型
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model