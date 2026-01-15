import multiprocessing as mp
import coverage
import os
import time

def setup_coverage(name):
    """为每个进程设置 coverage"""
    cov = coverage.Coverage(
        data_file=f'.coverage.{name}.{os.getpid()}',  # 每个进程使用唯一的数据文件
        branch=True,
        source=['your_source_directory'],  # 指定要收集覆盖率的源代码目录
    )
    cov.start()
    return cov

def child_of_child_process(name):
    """嵌套的子进程"""
    cov = setup_coverage(f"{name}_child")
    try:
        # 你的代码逻辑
        print(f"Nested child process {name} running with PID: {os.getpid()}")
        time.sleep(1)
    finally:
        cov.stop()
        cov.save()

def child_process(name):
    """子进程"""
    cov = setup_coverage(name)
    try:
        print(f"Child process {name} running with PID: {os.getpid()}")
        
        # 创建嵌套的子进程
        children = []
        for i in range(2):
            p = mp.Process(
                target=child_of_child_process,
                args=(f"{name}_{i}",)
            )
            children.append(p)
            p.start()
        
        # 等待所有嵌套子进程完成
        for p in children:
            p.join()
            
    finally:
        cov.stop()
        cov.save()

def main():
    """主进程"""
    # 设置主进程的 coverage
    main_cov = setup_coverage('main')
    
    try:
        print(f"Main process running with PID: {os.getpid()}")
        
        # 创建多个子进程
        processes = []
        for i in range(3):
            p = mp.Process(
                target=child_process,
                args=(f"process_{i}",)
            )
            processes.append(p)
            p.start()
        
        # 等待所有子进程完成
        for p in processes:
            p.join()
            
    finally:
        main_cov.stop()
        main_cov.save()
        
    # 合并所有覆盖率报告
    combine_coverage_reports()

def combine_coverage_reports():
    """合并所有进程的覆盖率报告"""
    cov = coverage.Coverage()
    cov.combine(data_files=[
        f for f in os.listdir('.')
        if f.startswith('.coverage.')
    ])
    
    # 生成报告
    cov.html_report(directory='coverage_report')
    cov.xml_report()
    
    # 清理临时文件
    for f in os.listdir('.'):
        if f.startswith('.coverage.'):
            os.remove(f)

if __name__ == '__main__':
    main()