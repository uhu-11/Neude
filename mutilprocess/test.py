import multiprocessing as mp
import coverage
import os
import time
import target, target2
import glob
import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys

TARGET_FILENAME = "target_file.txt"
PATH = 'watch'
class FileEventHandler(FileSystemEventHandler):
    def __init__(self, processes_to_terminate, observer, parent_pid):
        super().__init__()
        self.processes_to_terminate = processes_to_terminate
        self.observer=observer
        self.parent_pid = parent_pid

    def on_created(self, event):
        print(222222222222)
        file_name = os.path.basename(event.src_path)
        if file_name == TARGET_FILENAME:
            print(f"Detected target file: {file_name}")
            print(f"File {TARGET_FILENAME} created. Terminating processes...")
            for p in self.processes_to_terminate:
                if p.is_alive():
                    os.kill(p.pid, signal.SIGTERM)
            print("All processes terminated.")
            time.sleep(1)
            os.remove(PATH + '/' + TARGET_FILENAME)
            os.kill(self.parent_pid, signal.SIGINT)
            self.observer.stop()

def start_monitoring(processes_to_terminate, parent_pid):
    PATH = 'watch'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    observer = Observer()
    event_handler = FileEventHandler(processes_to_terminate, observer, parent_pid)
    
    observer.schedule(event_handler, path=PATH, recursive=False)
    print(333333333333333)
    return observer



def setup_coverage(name):
    """为每个进程设置 coverage"""
    cov = coverage.Coverage(
        data_file=f'/media/lzq/D/lzq/pylot_test/demo/.coverage.{name}.{os.getpid()}',  # 每个进程使用唯一的数据文件
        branch=True
    )
    return cov


def child_process(name, fuc, cov):
    """子进程"""
    # cov = setup_coverage(name)
    def signal_handler(signum, frame):
        print("child process received signal. Terminating child processes...")
        cov.stop()
        cov.save()
        print("Coverage data saved.")
        exit(0)
    print(111111111111111111)
    signal.signal(signal.SIGTERM, signal_handler)
    cov.start()
    try:
        fuc()
    except KeyboardInterrupt:
        print("child result has saved")

        # cov.stop()
        # cov.save()
        # raise KeyboardInterrupt

    
        

def main():
    """主进程"""
    # 设置主进程的 coverage
    print(233233233)
    main_cov = setup_coverage('main')
    main_cov.start()
    # signal.signal(signal.SIGINT, signal_handler)

    try:
        print(f"Main process running with PID: {os.getpid()}")
        cov = setup_coverage("process_0")
        p = mp.Process(
                target=child_process,
                args=(f"process_0",target.func, cov)
            )
        cov2 = setup_coverage("process_1")
        p2 = mp.Process(
                target=child_process,
                args=(f"process_0",target2.func, cov2)
            )
        
        print(123456789)
        observer = start_monitoring([p, p2], os.getpid())
        def sigint_handler(sig, frame):
            print('kill main')
            for pp in [p, p2]:
                # p.terminate()
                pp.join()
            # observer.stop()
            observer.join()
            main_cov.stop()
            main_cov.save()
            # 合并所有覆盖率报告
            combine_coverage_reports()
            sys.exit(0)
        signal.signal(signal.SIGINT, sigint_handler)

        p.start()
        p2.start()
        observer.start()
        # while True:
        #     time.sleep(1)
        # p.join()
        # p2.join()
        # observer.join()
        
    except KeyboardInterrupt:
        print("\nParent process is terminating the child process...")
        p.terminate()  # 终止子进程
        p2.terminate()  # 终止子进程
        p.join()  # 等待子进程结束
        p2.join()
        print("Child process terminated.")
 
    

        
    

def combine_coverage_reports():
    """合并所有进程的覆盖率报告"""
    cov = coverage.Coverage()
    
    # # 获取所有覆盖率数据文件
    coverage_files = glob.glob('.coverage.*')  # 假设覆盖率文件以 .coverage. 开头
    # print(coverage_files)

    # # 合并每个覆盖率数据文件
    cov.combine()  # 合并指定文件的覆盖率数据

    # 保存合并后的覆盖率数据
    cov.save()

    # 生成 HTML 报告
    
    with open("report.txt", "w") as file:
        cov.report(file=file)
    cov.html_report(directory='coverage_report')
    cov.xml_report()

    # # 清理临时文件
    # for coverage_file in coverage_files:
    #     os.remove(coverage_file)


if __name__ == '__main__':
    print('cccccccccccc')
    main()