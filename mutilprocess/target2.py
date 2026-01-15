import time
import random
import coverage
def func():
    print("Press Ctrl+C to stop the program.")
    
    count = 0  # 计数器
    status_message = "Running smoothly."  # 状态消息

    while count<5:
        # 每秒更新计数器
        count += 1
        
        # 生成一个随机数
        random_number = random.randint(1, 100)
        
        # 打印当前状态
        print(f"Count: {count}, Random Number: {random_number}, Status: {status_message}")
        
        # 每10次循环更新状态消息
        if count % 10 == 0:
            status_message = f"Still running... Count: {count}"
        
        time.sleep(1)  # 每秒循环一次
    
    # with open('files/file.txt', 'w') as f:
    #     f.write("Process completed successfully.\n")
    while True:
        pass
    

