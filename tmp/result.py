import json
import csv
import matplotlib.pyplot as plt
import numpy as np
def work_code_error(pos):
    newlst = [0]*100
    for p in pos:
        ind = p-1
        for i in range(ind, 100):
            newlst[i]+=1
    return newlst

def draw_obstacle(n):
    # 读取 JSON 文件
    exp1_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp1/exp1/100.json'  # 替换为你的 JSON 文件路径
    exp2_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp2/exp2/100.json'
    exp3_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp3/exp3/100.json'

    exp4_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp4/exp4/100.json'  # 替换为你的 JSON 文件路径
    exp5_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp5/exp5/100.json'
    exp6_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp6/exp6/100.json'

    exp7_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp7/exp7/100.json'  # 替换为你的 JSON 文件路径
    exp8_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp8/exp8/100.json'
    exp9_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp9/exp9/100.json'

    exp10_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp10/exp10/100.json'  # 替换为你的 JSON 文件路径
    exp11_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp11/exp11/100.json'
    exp12_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp12/exp12/100.json'

    # 并列
    with open(exp1_file_path, 'r', encoding='utf-8') as json_file:
        data1 = json.load(json_file)
    with open(exp2_file_path, 'r', encoding='utf-8') as json_file:
        data2 = json.load(json_file)
    with open(exp3_file_path, 'r', encoding='utf-8') as json_file:
        data3 = json.load(json_file)

    data1['code_errors'] = work_code_error([4,7])
    data2['code_errors'] = work_code_error([4,5,92])
    data3['code_errors'] = work_code_error([5,7,90])

    code_errors_binglie =[0]+[(a + b + c) / 3 for a, b, c in zip(data1['code_errors'], data2['code_errors'], data3['code_errors'])]
    model_errors_binglie = [0]+[(a + b + c) / 3 for a, b, c in zip(data1['model_errors'], data2['model_errors'], data3['model_errors'])]
    planning_errors_binglie = [0]+[(a + b + c) / 3 for a, b, c in zip(data1['planning_errors'], data2['planning_errors'], data3['planning_errors'])]
    control_errors_binglie = [0]+[(a + b + c) / 3 for a, b, c in zip(data1['control_errors'], data2['control_errors'], data3['control_errors'])]


    # 包含
    with open(exp4_file_path, 'r', encoding='utf-8') as json_file:
        data4 = json.load(json_file)
    with open(exp5_file_path, 'r', encoding='utf-8') as json_file:
        data5 = json.load(json_file)
    with open(exp6_file_path, 'r', encoding='utf-8') as json_file:
        data6 = json.load(json_file)

    data4['code_errors'] = work_code_error([5,7])
    data5['code_errors'] = work_code_error([5,11,65])
    data6['code_errors'] = work_code_error([4,10,87,99])

    code_errors_baohan = [0]+[(a + b + c) / 3 for a, b, c in zip(data4['code_errors'], data5['code_errors'], data6['code_errors'])]
    model_errors_baohan = [0]+[(a + b + c) / 3 for a, b, c in zip(data4['model_errors'], data5['model_errors'], data6['model_errors'])]
    planning_errors_baohan =[0]+[(a + b + c) / 3 for a, b, c in zip(data4['planning_errors'], data5['planning_errors'], data6['planning_errors'])]
    control_errors_baohan =[0]+ [(a + b + c) / 3 for a, b, c in zip(data4['control_errors'], data5['control_errors'], data6['control_errors'])]

    # nac
    with open(exp7_file_path, 'r', encoding='utf-8') as json_file:
        data7 = json.load(json_file)
    with open(exp8_file_path, 'r', encoding='utf-8') as json_file:
        data8 = json.load(json_file)
    with open(exp9_file_path, 'r', encoding='utf-8') as json_file:
        data9 = json.load(json_file)

    data7['code_errors'] = work_code_error([2,33,84,87])
    data8['code_errors'] = work_code_error([2])
    data9['code_errors'] = work_code_error([5])

  
    code_errors_nac =[0]+[(a + b + c) / 3 for a, b, c in zip(data7['code_errors'], data8['code_errors'], data9['code_errors'])]
    model_errors_nac =[0]+[(a + b + c) / 3 for a, b, c in zip(data7['model_errors'], data8['model_errors'], data9['model_errors'])]
    planning_errors_nac =[0]+[(a + b + c) / 3 for a, b, c in zip(data7['planning_errors'], data8['planning_errors'], data9['planning_errors'])]
    control_errors_nac = [0]+[(a + b + c) / 3 for a, b, c in zip(data7['control_errors'], data8['control_errors'], data9['control_errors'])]
    code_errors_nac=[0]*n
    planning_errors_nac=[0]*n
    control_errors_nac=[0]*n

    with open(exp10_file_path, 'r', encoding='utf-8') as json_file:
        data10 = json.load(json_file)
    with open(exp11_file_path, 'r', encoding='utf-8') as json_file:
        data11 = json.load(json_file)
    with open(exp12_file_path, 'r', encoding='utf-8') as json_file:
        data12 = json.load(json_file)

    data10['code_errors'] = work_code_error([2,12])
    data11['code_errors'] = work_code_error([7,35])
    data12['code_errors'] = work_code_error([4,13])

    code_errors_line =[0]+[(a + b + c) / 3 for a, b, c in zip(data10['code_errors'], data11['code_errors'], data12['code_errors'])]
    model_errors_line =[0]+[(a + b + c) / 3 for a, b, c in zip(data10['model_errors'], data11['model_errors'], data12['model_errors'])]
    planning_errors_line =[0]+[(a + b + c) / 3 for a, b, c in zip(data10['planning_errors'], data11['planning_errors'], data12['planning_errors'])]
    control_errors_line = [0]+[(a + b + c) / 3 for a, b, c in zip(data10['control_errors'], data11['control_errors'], data12['control_errors'])]
    model_errors_line=[0]*n
    planning_errors_line=[0]*n
    control_errors_line=[0]*n

    draw_coverage("Obstacle Detection System Errors", code_errors_nac, code_errors_line, code_errors_binglie, code_errors_baohan, "/home/lzq/test_reslut/会议论文实验结果/错误图/obstacle_system_from0.png")
    draw_coverage("Obstacle Detection Perception Errors", model_errors_nac, model_errors_line, model_errors_binglie, model_errors_baohan, "/home/lzq/test_reslut/会议论文实验结果/错误图/obstacle_model_from0.png")
    draw_coverage("Obstacle Detection Plannning Errors", planning_errors_nac, planning_errors_line, planning_errors_binglie, planning_errors_baohan, "/home/lzq/test_reslut/会议论文实验结果/错误图/obstacle_planning_from0.png")
    draw_coverage("Obstacle Detection Control Errors", control_errors_nac, control_errors_line, control_errors_binglie, control_errors_baohan, "/home/lzq/test_reslut/会议论文实验结果/错误图/obstacle_control_from0.png")

def draw_traffic(n):
    # 读取 JSON 文件
    exp1_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp13/exp13/100.json'  # 替换为你的 JSON 文件路径
    exp2_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp14/exp14/100.json'
    exp3_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp15/exp15/100.json'

    exp4_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp16/exp16/100.json'  # 替换为你的 JSON 文件路径
    exp5_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp17/exp17/100.json'
    exp6_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp18/exp18/100.json'

    exp7_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp19/exp19/100.json'  # 替换为你的 JSON 文件路径
    exp8_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp20/exp20/100.json'
    exp9_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp21/exp21/100.json'

    exp10_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp22/exp22/100.json'  # 替换为你的 JSON 文件路径
    exp11_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp23/exp23/100.json'
    exp12_file_path = '/media/lzq/My Passport/pylot实验数据/实验结果/exp24/exp24/100.json'

    # 并列
    with open(exp1_file_path, 'r', encoding='utf-8') as json_file:
        data1 = json.load(json_file)
    with open(exp2_file_path, 'r', encoding='utf-8') as json_file:
        data2 = json.load(json_file)
    with open(exp3_file_path, 'r', encoding='utf-8') as json_file:
        data3 = json.load(json_file)

    data1['code_errors'] = work_code_error([3])
    data2['code_errors'] = work_code_error([3])
    data3['code_errors'] = work_code_error([4,89])


    code_errors_binglie =[0]+[(a + b + c) / 3 for a, b, c in zip(data1['code_errors'], data2['code_errors'], data3['code_errors'])]
    model_errors_binglie = [0]+[(a + b + c) / 3 for a, b, c in zip(data1['model_errors'], data2['model_errors'], data3['model_errors'])]
    planning_errors_binglie = [0]+[(a + b + c) / 3 for a, b, c in zip(data1['planning_errors'], data2['planning_errors'], data3['planning_errors'])]
    control_errors_binglie =[0]+[(a + b + c) / 3 for a, b, c in zip(data1['control_errors'], data2['control_errors'], data3['control_errors'])]
    

    # 包含
    with open(exp4_file_path, 'r', encoding='utf-8') as json_file:
        data4 = json.load(json_file)
    with open(exp5_file_path, 'r', encoding='utf-8') as json_file:
        data5 = json.load(json_file)
    with open(exp6_file_path, 'r', encoding='utf-8') as json_file:
        data6 = json.load(json_file)

    data4['code_errors'] = work_code_error([4])
    data5['code_errors'] = work_code_error([4])
    data6['code_errors'] = work_code_error([5])


    code_errors_baohan =[0]+[(a + b + c) / 3 for a, b, c in zip(data4['code_errors'], data5['code_errors'], data6['code_errors'])]
    model_errors_baohan =[0]+[(a + b + c) / 3 for a, b, c in zip(data4['model_errors'], data5['model_errors'], data6['model_errors'])]
    planning_errors_baohan = [0]+[(a + b + c) / 3 for a, b, c in zip(data4['planning_errors'], data5['planning_errors'], data6['planning_errors'])]
    control_errors_baohan =[0]+[(a + b + c) / 3 for a, b, c in zip(data4['control_errors'], data5['control_errors'], data6['control_errors'])]

    # nac
    with open(exp7_file_path, 'r', encoding='utf-8') as json_file:
        data7 = json.load(json_file)
    with open(exp8_file_path, 'r', encoding='utf-8') as json_file:
        data8 = json.load(json_file)
    with open(exp9_file_path, 'r', encoding='utf-8') as json_file:
        data9 = json.load(json_file)

    data7['code_errors'] = work_code_error([2])
    data8['code_errors'] = work_code_error([3])
    data9['code_errors'] = work_code_error([3])

  
    code_errors_nac =[0]+[(a + b + c) / 3 for a, b, c in zip(data7['code_errors'], data8['code_errors'], data9['code_errors'])]
    model_errors_nac = [0]+[(a + b + c) / 3 for a, b, c in zip(data7['model_errors'], data8['model_errors'], data9['model_errors'])]
    planning_errors_nac = [0]+[(a + b + c) / 3 for a, b, c in zip(data7['planning_errors'], data8['planning_errors'], data9['planning_errors'])]
    control_errors_nac = [0]+[(a + b + c) / 3 for a, b, c in zip(data7['control_errors'], data8['control_errors'], data9['control_errors'])]
    code_errors_nac=[0]*n
    planning_errors_nac=[0]*n
    control_errors_nac=[0]*n

    # 行
    with open(exp10_file_path, 'r', encoding='utf-8') as json_file:
        data10 = json.load(json_file)
    with open(exp11_file_path, 'r', encoding='utf-8') as json_file:
        data11 = json.load(json_file)
    with open(exp12_file_path, 'r', encoding='utf-8') as json_file:
        data12 = json.load(json_file)

    data10['code_errors'] = work_code_error([5,91,93])
    data11['code_errors'] = work_code_error([5,80,93])
    data12['code_errors'] = work_code_error([6])

    code_errors_line =[0]+[(a + b + c) / 3 for a, b, c in zip(data10['code_errors'], data11['code_errors'], data12['code_errors'])]
    model_errors_line =[0]+[(a + b + c) / 3 for a, b, c in zip(data10['model_errors'], data11['model_errors'], data12['model_errors'])]
    planning_errors_line =[0]+[(a + b + c) / 3 for a, b, c in zip(data10['planning_errors'], data11['planning_errors'], data12['planning_errors'])]
    control_errors_line =[0]+[(a + b + c) / 3 for a, b, c in zip(data10['control_errors'], data11['control_errors'], data12['control_errors'])]
    model_errors_line=[0]*n
    planning_errors_line=[0]*n
    control_errors_line=[0]*n

    draw_coverage("Traffic Light Detection System Errors", code_errors_nac, code_errors_line, code_errors_binglie, code_errors_baohan, "/home/lzq/test_reslut/会议论文实验结果/错误图/tl_system_from0.png")
    draw_coverage("Traffic Light Detection Perception Errors", model_errors_nac, model_errors_line, model_errors_binglie, model_errors_baohan, "/home/lzq/test_reslut/会议论文实验结果/错误图/tl_model_from0.png")
    draw_coverage("Traffic Light Detection Plannning Errors", planning_errors_nac, planning_errors_line, planning_errors_binglie, planning_errors_baohan, "/home/lzq/test_reslut/会议论文实验结果/错误图/tl_planning_from0.png")
    draw_coverage("Traffic Light Detection Control Errors", control_errors_nac, control_errors_line, control_errors_binglie, control_errors_baohan, "/home/lzq/test_reslut/会议论文实验结果/错误图/tl_control_from0.png")

def draw_coverage(title, y1, y2, y3, y4, path):
    # x 轴从 1 到 len(y)

    x = np.arange(0, 101)
    # pchip_interp = PchipInterpolator(x, y1)
    # pchip_interp2 = PchipInterpolator(x, y2)
    # pchip_interp3 = PchipInterpolator(x, y3)
    # x_smooth = np.linspace(1, len(y1), 300)  # 生成平滑 x 轴数据
    # y1_smooth = pchip_interp(x_smooth)  # 计算平滑 y 值
    # y2_smooth = pchip_interp2(x_smooth)
    # y3_smooth = pchip_interp3(x_smooth)

    plt.plot(x, y1, linestyle='-', color='b', label='DeepHunter')
    plt.plot(x, y2, linestyle='--', color='r', label='PythonFuzz')
    plt.plot(x, y3, linestyle='dotted', color='g', label='Global Weight(Ours)')
    plt.plot(x, y4, linestyle='-.', color='y', label='Path-Neuron(Ours)')
  

    # 添加标题和轴标签
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Crash")
    # plt.ylim(0.2, 0.7)

    # 显示图例
    plt.legend(loc='best',framealpha=0.5)
    plt.grid(alpha=0.5) 

        # 保存图像，不显示
    plt.savefig(path, dpi=300, bbox_inches='tight')
    # 关闭图像，避免占用内存
    plt.close()

if __name__ == '__main__':
    draw_traffic(101)
    draw_obstacle(101)
    print('画完了')