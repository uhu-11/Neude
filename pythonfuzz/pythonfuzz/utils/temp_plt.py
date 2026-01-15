import matplotlib.pyplot as plt
import json
from scipy.interpolate import PchipInterpolator
import numpy as np
names = {'Mix Coverage Rate','Line Coverage Rate','Code Crash Number','Model Crash Number'}
def draw_cov_img(y):
    # x 轴从 1 到 len(y)

    x = np.arange(1, len(y) + 1)
    pchip_interp = PchipInterpolator(x, y)
    x_smooth = np.linspace(1, len(y), 300)  # 生成平滑 x 轴数据
    y_smooth = pchip_interp(x_smooth)  # 计算平滑 y 值

    plt.plot(x_smooth, y_smooth, linestyle='-', color='#4E616C')
  

    # 添加标题和轴标签
    plt.title("Mix Coverage Rate Variation Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Mix Coverage Rate")

    # 显示图例
    # plt.legend()
    plt.grid(alpha=0.5) 

        # 保存图像，不显示
    plt.savefig(f"/home/lzq/test_reslut/trafficlight4/result/every_cov_img/mix_cov_vv.png", dpi=300, bbox_inches='tight')
    # 关闭图像，避免占用内存
    plt.close()


def draw_crash_img(y):
    # x 轴从 1 到 len(y)

    x = np.arange(1, len(y) + 1)
    pchip_interp = PchipInterpolator(x, y)
    x_smooth = np.linspace(1, len(y), 300)  # 生成平滑 x 轴数据
    y_smooth = pchip_interp(x_smooth)  # 计算平滑 y 值

    plt.plot(x_smooth, y_smooth, linestyle='-', color='#4E616C')
  

    # 添加标题和轴标签
    plt.title("Model Crash Number Variation Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("crash")

    # 显示图例
    # plt.legend()
    plt.grid(alpha=0.5) 

    # 保存图像，不显示
    plt.savefig(f"/home/lzq/test_reslut/pillow4/result/every_crash_img/model_crash_100.png", dpi=300, bbox_inches='tight')
    # 关闭图像，避免占用内存
    plt.close()



def draw_coverage(y1, y2, y3):
    # x 轴从 1 到 len(y)

    x = np.arange(1, len(y1) + 1)
    pchip_interp = PchipInterpolator(x, y1)
    pchip_interp2 = PchipInterpolator(x, y2)
    pchip_interp3 = PchipInterpolator(x, y3)
    x_smooth = np.linspace(1, len(y1), 300)  # 生成平滑 x 轴数据
    y1_smooth = pchip_interp(x_smooth)  # 计算平滑 y 值
    y2_smooth = pchip_interp2(x_smooth)
    y3_smooth = pchip_interp3(x_smooth)

    plt.plot(x_smooth, y1_smooth, linestyle='-', color='b', label='line')
    plt.plot(x_smooth, y2_smooth, linestyle='--', color='r', label='nac')
    plt.plot(x_smooth, y3_smooth, linestyle='dotted', color='g', label='mix')
  

    # 添加标题和轴标签
    plt.title("Coverage Rate Variation Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Coverage Rate")
    plt.ylim(0.2, 0.7)

    # 显示图例
    plt.legend(loc='best',framealpha=0.5)
    plt.grid(alpha=0.5) 

        # 保存图像，不显示
    plt.savefig(f"/home/lzq/test_reslut/trafficlight4/result/every_cov_img/cov_50.png", dpi=300, bbox_inches='tight')
    # 关闭图像，避免占用内存
    plt.close()




def draw_error(y1, y2):
    # x 轴从 1 到 len(y)

    x = np.arange(1, len(y1) + 1)
    pchip_interp = PchipInterpolator(x, y1)
    pchip_interp2 = PchipInterpolator(x, y2)
    x_smooth = np.linspace(1, len(y1), 300)  # 生成平滑 x 轴数据
    y1_smooth = pchip_interp(x_smooth)  # 计算平滑 y 值
    y2_smooth = pchip_interp2(x_smooth)

    plt.plot(x_smooth, y1_smooth, linestyle='-', color='b', label='code')
    plt.plot(x_smooth, y2_smooth, linestyle='--', color='r', label='model')
  

    # 添加标题和轴标签
    plt.title("Crash Number Variation Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("crash")

    # 显示图例
    plt.legend(loc='best',framealpha=0.5)
    plt.grid(alpha=0.5) 


        # 保存图像，不显示
    plt.savefig(f"/home/lzq/test_reslut/pillow4/result/every_crash_img/crash_100.png", dpi=300, bbox_inches='tight')
    # 关闭图像，避免占用内存
    plt.close()

# if __name__ =='__main__':
#     with open('/home/lzq/test_reslut/obstacles2/result/datas/50.json', 'r', encoding='utf-8') as f:
#         data = json.load(f)  # 将 JSON 数据加载为 Python 对象
#     # data['code_errors'] = [0 for i in data['code_errors']]
#     print(data['plt_nac_cov'][48],data['plt_nac_cov'][49])
#     data['plt_nac_cov'].pop(0) 
#     data['plt_nac_cov'].append(data['plt_nac_cov'][48])

#     with open('/home/lzq/test_reslut/obstacle/result/datas/50.json', 'r', encoding='utf-8') as f:
#         data2 = json.load(f)
#     data['plt_line_cov']=data2['plt_line_cov']
#     data['plt_combine_cov']=[(x * 0.45) + (y * 0.55) for x, y in zip(data['plt_line_cov'], data['plt_nac_cov'])]

#     # draw_cov_img(data['plt_combine_cov'])
#     draw_coverage(data['plt_line_cov'],data['plt_nac_cov'],data['plt_combine_cov'])
#     print(data.keys())

if __name__ =='__main__':
    with open('/home/lzq/test_reslut/trafficlight4/result/datas/50.json', 'r', encoding='utf-8') as f:
        data = json.load(f)  # 将 JSON 数据加载为 Python 对象
    # data['code_errors'] = [0 for i in data['code_errors']]
    print(data['plt_nac_cov'][48],data['plt_nac_cov'][49])


    with open('/home/lzq/test_reslut/trafficlight3/result/datas/50.json', 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    data['plt_line_cov']=data2['plt_line_cov']
    data['plt_combine_cov']=[(x * 0.45) + (y * 0.55) for x, y in zip(data['plt_line_cov'], data['plt_nac_cov'])]

    draw_cov_img(data['plt_combine_cov'])
    # draw_coverage(data['plt_line_cov'],data['plt_nac_cov'],data['plt_combine_cov'])
    print(data.keys())