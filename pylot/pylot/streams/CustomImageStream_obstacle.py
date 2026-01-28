import os
import cv2
import time
import sys
import erdos
from multiprocessing import Process
import logging
import subprocess
import psutil
import signal
import pylot.flags
from absl import app, flags
from pylot.perception.messages import FrameMessage
from pylot.perception.camera_frame import CameraFrame
import pickle
import numpy as np

from pylot.drivers.sensor_setup import RGBCameraSetup
from pylot.inputs import images
from pylot.simulation.utils import set_asynchronous_mode
neude_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'neude', 'neude')  # 修改为你的 neude 实际路径
sys.path.append(neude_path)



class CustomImageStream_obstacle(erdos.Operator):
    @staticmethod
    def connect(center_camera_stream):
        obstacle_stream = erdos.WriteStream()
        notify_reading_stream = erdos.WriteStream()
        print("Connect Obstacle Detection Stream:", obstacle_stream)
        return [obstacle_stream, notify_reading_stream]

    def __init__(self, center_camera_stream: erdos.ReadStream,
                 obstacle_stream: erdos.WriteStream,
                 notify_reading_stream: erdos.WriteStream,
                 image_folder: str, center_camera_setup, 
                 fps=30):
        super().__init__()
        try:
            logging.warning("Initializing ObstacleDetectionStream")
            self.obstacle_stream = obstacle_stream
            self.center_camera_stream = center_camera_stream
            self.fps = fps
            self.interval = 1.0 / fps
            self.image_folder = image_folder
            self.image_files = sorted(os.listdir(image_folder))  # 获取图像文件列表
            self.current_index = 0
            self.center_camera_setup = center_camera_setup
            
            # 添加回调
            center_camera_stream.add_callback(self.on_frame, [obstacle_stream])
        except Exception as e :
            print(f"Error in ObstacleDetectionStream initialization: {e}")

    # def on_frame(self, msgs: erdos.Message, obstacle_stream: erdos.WriteStream):
    #     # 从本地文件夹加载图像
    #     if self.current_index < len(self.image_files):
    #         image_path = os.path.join(self.image_folder, self.image_files[self.current_index])
    #         image = cv2.imread(image_path)  # 读取图像
    #         if image is not None:
    #             # 直接发送读取的图像到 obstacle_stream
    #             self.width = image.shape[1]
    #             self.height = image.shape[0]
                
    #             timestamp = erdos.Timestamp(coordinates=[self.current_index])  # 创建时间戳
    #             tl_camera_setup = RGBCameraSetup('traffic_light_camera',
    #                                 self.width,
    #                                 self.height, self.tl_transform,fov=45)
    #             msg = FrameMessage(timestamp, CameraFrame(image, 'BGR', tl_camera_setup))  # 创建消息，不使用 tl_transform
    #             obstacle_stream.send(msg)  # 发送图像消息
    #             logging.warning(f"Sent image: {image_path}")
    #             # obstacle_stream.send(erdos.WatermarkMessage(timestamp)) 
    #             self.current_index += 1  # 更新索引
    #         else:
    #             logging.warning(f"Failed to read image: {image_path}")
    #     else:
    #         os.kill(os.getppid(), signal.SIGINT)

    def get_next_image(self):
        if self.current_index < len(self.image_files):
            image_path = os.path.join(self.image_folder, self.image_files[self.current_index])
            image = cv2.imread(image_path)
            # print(image.shape)
            # np.save('/media/lzq/D/lzq/pylot_test/img_ind.npy', np.array(self.current_index))
            self.current_index += 1
            return image
        else:
            return None  # 没有更多图像
    def on_frame(self, msgs: erdos.Message,
                 camera_stream: erdos.WriteStream):
  

        if self.current_index < len(self.image_files):
            image = self.get_next_image()
            if image is not None:
                # print("读取的图片形状：", image.shape)
                self.width = image.shape[1]
                self.height = image.shape[0]
               
                timestamp = erdos.Timestamp(coordinates=[self.current_index])  # 创建时间戳
                msg = FrameMessage(timestamp, CameraFrame(image, 'BGR', self.center_camera_setup))  # 创建消息
                logging.warning("send msg")
                self.obstacle_stream.send(msg)
                start_time = time.time()  # 发送图像消息
                self.obstacle_stream.send(erdos.WatermarkMessage(timestamp)) 
        else:
            os.kill(os.getppid(), signal.SIGINT)