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

from pylot.drivers.sensor_setup import RGBCameraSetup
from pylot.inputs import images
from pylot.simulation.utils import set_asynchronous_mode
pythonfuzz_path = "/media/lzq/D/lzq/pylot_test/pythonfuzz"  # 修改为你的 pythonfuzz 实际路径
sys.path.append(pythonfuzz_path)


class CustomImageStream(erdos.Operator):
    @staticmethod
    def connect(tl_camera_stream):
        camera_stream = erdos.WriteStream()
        notify_reading_stream = erdos.WriteStream()
        print("Connect Custom Stream:", camera_stream)
        return [camera_stream, notify_reading_stream]

    def __init__(self, tl_camera_stream: erdos.ReadStream,
                 camera_stream: erdos.WriteStream,
                 notify_reading_stream: erdos.WriteStream,
                 image_folder, tl_camera_setup, fps=30):
        print('CustomImageStream init .........')
        super().__init__() 
        try:
            logging.warning("Initializing CustomImageStream")
            self.image_folder = image_folder
            self.image_files = sorted(os.listdir(image_folder))
            self.current_index = 0
            self.fps = fps
            self.camera_stream = camera_stream
            print('self.camera_stream',self.camera_stream)
            self.interval = 1.0 / fps
            self.on_frame_callback = None
            self.width = 0
            self.height = 0
            self.tl_camera_setup = tl_camera_setup
            tl_camera_stream.add_callback(self.on_frame, [camera_stream])
        except Exception as e:
       
            with open('/media/lzq/D/lzq/pylot_test/error_log.txt', 'w') as f:
                f.write("error:"+str(e))
            os.kill(os.getppid(), signal.SIGINT)
    def get_next_image(self):
        if self.current_index < len(self.image_files):
            image_path = os.path.join(self.image_folder, self.image_files[self.current_index])
            image = cv2.imread(image_path)
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
                # tl_camera_setup = RGBCameraSetup('traffic_light_camera',
                #                     self.width,
                #                     self.height,
                #                     self.tl_transform, 45)
                timestamp = erdos.Timestamp(coordinates=[self.current_index])  # 创建时间戳
                msg = FrameMessage(timestamp, CameraFrame(image, 'BGR', self.tl_camera_setup))  # 创建消息
                logging.warning("send msg")
                self.camera_stream.send(msg)
                start_time = time.time()  # 发送图像消息
                self.camera_stream.send(erdos.WatermarkMessage(timestamp)) 
        else:
            os.kill(os.getppid(), signal.SIGINT)


    # def run(self):
    #     # print("Running CustomImageStream")
    #     try:
    #     with open('/media/lzq/D/lzq/pylot_test/mutilprocess/watch/target_file.txt', 'w') as f:
    #             f.write("Process completed successfully.\n")
    #     while self.current_index < len(self.image_files):
    #         image = self.get_next_image()
    #         if image is not None:
    #             print("读取的图片形状：", image.shape)
    #             self.width = image.shape[1]
    #             self.height = image.shape[0]
    #             tl_camera_setup = RGBCameraSetup('traffic_light_camera',
    #                                 self.width,
    #                                 self.height,
    #                                 self.tl_transform, 45)
    #             timestamp = erdos.Timestamp(coordinates=[self.current_index])  # 创建时间戳
    #             msg = FrameMessage(timestamp, CameraFrame(image, 'BGR', tl_camera_setup))  # 创建消息
    #             logging.warning("send msg")
    #             self.camera_stream.send(msg)
    #             start_time = time.time()  # 发送图像消息
    #             self.camera_stream.send(erdos.WatermarkMessage(timestamp)) 
            
    #         # 等待下一个捕获时间
    #         elapsed_time = time.time() - start_time
    #         time_to_wait = self.interval - elapsed_time
    #         print('time_to_wait:', time_to_wait)
    #         if time_to_wait > 0 and self.current_index < len(self.image_files)-1:
    #             time.sleep(time_to_wait)
    #     # 在循环结束后终止进程
    #     # print("Terminating CustomImageStream process.ooooooooooooooooo")

    #     # with open('/media/lzq/D/lzq/pylot_test/mutilprocess/watch/target_file.txt', 'w') as f:
    #     #     f.write("Process completed successfully.\n")
    #     # print("has created")
    #     #获取父进程并发送SIGINT信号
    #     os.kill(os.getppid(), signal.SIGINT)

    #     # current_pid = os.getpid()

    #     # with open('/media/lzq/D/lzq/pylot_test/tmp/child_pid.txt', 'r') as f:
    #     #     child_pid = int(f.read())

    #     # print("childid:", child_pid)
    #     # kill_process_tree(child_pid, exclude_pid=current_pid)   
    #     # os.kill(current_pid, signal.SIGINT)
        
    #     # except KeyboardInterrupt:
    #     #     p = Process(target=process_function, args=('python fuzz.py',))
    #     #     p.start()  # 启动进程
    #     #     p.join() 
    #     #     logging.warning("KeyboardInterrupt received. Ts.")


def kill_process_tree(pid, exclude_pid=None):
    print(f"Killing process tree with PID: {pid}")
    """
    递归终止进程树中的所有进程，可以排除指定的进程ID
    
    Args:
        pid: 要终止的进程ID
        exclude_pid: 要排除的进程ID（通常是当前进程）
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # 按照进程树的层级顺序终止进程
        for child in reversed(children):
            if child.pid != exclude_pid:  # 跳过当前进程
                try:
                    print(f"Killing child process: {child.pid}")
                    os.kill(child.pid, signal.SIGUSR1)
                except (psutil.NoSuchProcess, OSError):
                    pass
        
        # 如果父进程不是被排除的进程，才终止它
        # if pid != exclude_pid:
        #     print(f"Killing parent process: {pid}")
        #     os.kill(pid, signal.SIGKILL)
            
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found")
        pass

# 使用示例


def shutdown_pylot(node_handle, client, world):
    if node_handle:
        node_handle.shutdown()
    else:
        print('WARNING: The Pylot dataflow failed to initialize.')
    FLAGS = flags.FLAGS
    if FLAGS.simulation_recording_file is not None:
        client.stop_recorder()
    set_asynchronous_mode(world)
    if pylot.flags.must_visualize():
        import pygame
        pygame.quit()
    
def kill_processes_by_command(command):
    # 使用 ps 命令获取所有进程
    try:
        # 执行 ps aux 命令
        i=0
        ps_output = subprocess.check_output(['ps', 'aux'], text=True)
        # 遍历每一行，查找包含指定命令的进程
        for line in ps_output.splitlines():
            if command in line and 'grep' not in line:  # 排除 grep 进程
                if i<2:
                    i+=1
                    continue
                # 提取 PID
                pid = int(line.split()[1])  # 第二列是 PID
                os.kill(pid, signal.SIGKILL)  # 使用 SIGKILL 杀掉进程
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

def process_function(command):
    # 等待 100 毫秒
    time.sleep(1)
    # 再次调用
    kill_processes_by_command(command)

