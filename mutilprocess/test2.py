from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import multiprocessing as mp
from target import func
import os, signal
from test import main
import sys

def child_process(name, fuc):
    """子进程"""
    # cov = setup_coverage(name)
    def handler(sig, frame):
        print("kill child")
        sys.exit(0)
    signal.signal(signal.SIGUSR1, handler)
    try:
        fuc()
    except KeyboardInterrupt:
        print("child result has saved")

if __name__ == '__main__':
    p = mp.Process(
                    target=child_process,
                    args=(f"process_0",func)
                )
    p.start()
    time.sleep(3)

    os.kill(p.pid, signal.SIGUSR1)
    p.join()