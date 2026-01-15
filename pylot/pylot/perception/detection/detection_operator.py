"""Implements an operator that detects obstacles."""
import logging
import time

import erdos

import numpy as np

import pylot.utils
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D, \
    OBSTACLE_LABELS, load_coco_bbox_colors, load_coco_labels
from pylot.perception.messages import ObstaclesMessage
from pylot.global_var import COV_FILE_PATH
import tensorflow as tf
import coverage
import multiprocessing as mp
import signal
import os
from pylot.coverage_decorator import coverage_decorator

class DetectionOperator(erdos.Operator):
    """Detects obstacles using a TensorFlow model.

    The operator receives frames on a camera stream, and runs a model for each
    frame.

    Args:
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends
            :py:class:`~pylot.perception.messages.ObstaclesMessage` messages.
        model_path(:obj:`str`): Path to the model pb file.
        flags (absl.flags): Object to be used to access absl flags.
    """
    @coverage_decorator
    def __init__(self, camera_stream: erdos.ReadStream,
                 time_to_decision_stream: erdos.ReadStream,
                 obstacles_stream: erdos.WriteStream, model_path: str, flags):
        camera_stream.add_callback(self.on_msg_camera_stream,
                                   [obstacles_stream])
        time_to_decision_stream.add_callback(self.on_time_to_decision_update)
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._obstacles_stream = obstacles_stream
        self._iter=0

        pylot.utils.set_tf_loglevel(logging.ERROR)

        # Only sets memory growth for flagged GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(
            [physical_devices[self._flags.obstacle_detection_gpu_index]],
            'GPU')
        tf.config.experimental.set_memory_growth(
            physical_devices[self._flags.obstacle_detection_gpu_index], True)

        # Load the model from the saved_model format file.
        self._model = tf.saved_model.load(model_path)

        self._coco_labels = load_coco_labels(self._flags.path_coco_labels)
        # print('self._flags.path_coco_labels:%s', self._flags.path_coco_labels)
        logging.warning('self._flags.path_coco_labels:%s', self._flags.path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)
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
            :py:class:`~pylot.perception.messages.ObstaclesMessage` messages.
        """
        obstacles_stream = erdos.WriteStream()
        return [obstacles_stream]
    @coverage_decorator
    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))
        # Sending top watermark because the operator is not flowing
        # watermarks.
        self._obstacles_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
    @coverage_decorator
    def on_time_to_decision_update(self, msg: erdos.Message):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))

    @erdos.profile_method()
    @coverage_decorator
    def on_msg_camera_stream(self, msg: erdos.Message,
                             obstacles_stream: erdos.WriteStream):
        """Invoked whenever a frame message is received on the stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.FrameMessage`): Message
                received.
            obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which
                the operator sends
                :py:class:`~pylot.perception.messages.ObstaclesMessage`
                messages.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        start_time = time.time()
        # The models expect BGR images.
        assert msg.frame.encoding == 'BGR', 'Expects BGR frames'
        num_detections, res_boxes, res_scores, res_classes = self.__run_model(
            msg.frame.frame)
        # logging.warning('num_detections:%s',num_detections)
        # logging.warning('res_boxes:%s',res_boxes)
        # logging.warning('res_scores:%s',res_scores)
        # logging.warning('res_classes:%s',res_classes)

        transformed_boxes = []
        for box in res_boxes:
            transformed_box = [
                box[1] * msg.frame.camera_setup.width,    # x1
                box[0] * msg.frame.camera_setup.height,   # y1
                box[3] * msg.frame.camera_setup.width,    # x2
                box[2] * msg.frame.camera_setup.height    # y2
            ]
            transformed_boxes.append(transformed_box)
        transformed_boxes = [[float(tensor.numpy()) for tensor in sublist] for sublist in transformed_boxes]
        res_map = {'labels':res_classes, 'boxes' : transformed_boxes}
   
        np.save(f'/media/lzq/D/lzq/pylot_test/pylot/predictions/obstacle_trafficlight/prediction_{self._iter:08d}.npy', res_map)
        self._iter+=1
        print('障碍物保存npy完毕.......')


        obstacles = []
        for i in range(0, num_detections):
            if res_classes[i] in self._coco_labels:
                if (res_scores[i] >=
                        self._flags.obstacle_detection_min_score_threshold):
                    transformed_box = [
                        res_boxes[i][1] * msg.frame.camera_setup.width,    # x1
                        res_boxes[i][0] * msg.frame.camera_setup.height,   # y1
                        res_boxes[i][3] * msg.frame.camera_setup.width,    # x2
                        res_boxes[i][2] * msg.frame.camera_setup.height    # y2
                    ]
                    # print(f"{self._iter}轮检测到的对象:{self._coco_labels[res_classes[i]]},分数：{res_scores[i]},识别框：{transformed_box}")
                    if (self._coco_labels[res_classes[i]] in OBSTACLE_LABELS):
                        obstacles.append(
                            Obstacle(BoundingBox2D(
                                int(res_boxes[i][1] *
                                    msg.frame.camera_setup.width),
                                int(res_boxes[i][3] *
                                    msg.frame.camera_setup.width),
                                int(res_boxes[i][0] *
                                    msg.frame.camera_setup.height),
                                int(res_boxes[i][2] *
                                    msg.frame.camera_setup.height)),
                                     res_scores[i],
                                     self._coco_labels[res_classes[i]],
                                     id=self._unique_id))
                        self._unique_id += 1
                    else:
                        self._logger.warning(
                            'Ignoring non essential detection {}'.format(
                                self._coco_labels[res_classes[i]]))
            else:
                self._logger.warning('Filtering unknown class: {}'.format(
                    res_classes[i]))

        self._logger.debug('@{}: {} obstacles: {}'.format(
            msg.timestamp, self.config.name, obstacles))

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        # Send out obstacles.
        obstacles_stream.send(
            ObstaclesMessage(msg.timestamp, obstacles, runtime))
        obstacles_stream.send(erdos.WatermarkMessage(msg.timestamp))

        if self._flags.log_detector_output:
            msg.frame.annotate_with_bounding_boxes(msg.timestamp, obstacles,
                                                   None, self._bbox_colors)
            msg.frame.save(msg.timestamp.coordinates[0], self._flags.data_path,
                           'detector-{}'.format(self.config.name))
    @coverage_decorator
    def __run_model(self, image_np):
        # Expand dimensions since the model expects images to have
        # shape: [1, None, None, 3]

        image_np_expanded = np.expand_dims(image_np, axis=0)

        infer = self._model.signatures['serving_default']
        result = infer(tf.convert_to_tensor(value=image_np_expanded))

        boxes = result['boxes']
        scores = result['scores']
        classes = result['classes']
        num_detections = result['detections']

        num_detections = int(num_detections[0])
        res_classes = [int(cls) for cls in classes[0][:num_detections]]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]
        # print(res_scores)
        return num_detections, res_boxes, res_scores, res_classes
