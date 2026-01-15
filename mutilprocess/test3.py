from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import multiprocessing as mp
from target import func
import os

# 目标文件名
TARGET_FILENAME = "target_file.txt"
# 监视的文件夹路径
WATCH_DIRECTORY = "watch"

# 处理目标文件的函数
def process_file(file_path):
    print(f"Processing file: {file_path}")
    # 在这里添加处理逻辑

# 事件处理器
class FileCreateHandler(FileSystemEventHandler):
    def __init__(self,observer):
        super().__init__()
        self.observer = observer
    def on_created(self, event):
        if not event.is_directory:  # 确保是文件而不是文件夹
            file_name = os.path.basename(event.src_path)
            if file_name == TARGET_FILENAME:
                print(f"Detected target file: {file_name}")
                process_file(event.src_path)
                self.observer.stop()

# 监视文件夹
def start_watching(directory):
    observer = Observer()
    event_handler = FileCreateHandler(observer)
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    try:
        p = mp.Process(
                target=func,
            ) # 保持主线程运行
        p.start()
        p.join()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()



if __name__ == "__main__":
    start_watching(WATCH_DIRECTORY)
