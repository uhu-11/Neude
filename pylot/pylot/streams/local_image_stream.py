# pylot/pylot/streams/local_image_stream.py
import os
import cv2
import erdos
from pylot.perception.messages import FrameMessage
from pylot.perception.camera_frame import CameraFrame

# pylot/pylot/streams/local_image_stream.py
class LocalImageStream(erdos.Operator):
    def __init__(self, image_folder: str, camera_stream: erdos.WriteStream):
        super().__init__()
        self.image_folder = image_folder
        self.camera_stream = camera_stream
        self.image_files = sorted(os.listdir(image_folder))
        self.current_index = 0

    @staticmethod
    def connect():
        camera_stream = erdos.WriteStream()  # 只创建输出流
        return [camera_stream]

    def run(self):
        while self.current_index < len(self.image_files):
            image_path = os.path.join(self.image_folder, self.image_files[self.current_index])
            image = cv2.imread(image_path)
            if image is not None:
                timestamp = erdos.Timestamp(coordinates=[self.current_index])
                msg = FrameMessage(timestamp, CameraFrame(image, 'BGR', None))  # None for camera setup
                self.camera_stream.send(msg)
                self.current_index += 1
            else:
                print(f"Error reading image: {image_path}")