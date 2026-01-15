import matplotlib.pyplot as plt
import os
# 示例 y 轴数据（范围在 0 到 1 之间）
y_line1 = [12, 13, 15, 17, 19, 18, 16, 14]
y_nac1 = [12, 14, 16, 18, 11, 19, 17, 15]
y_nac2 = [21, 15, 16, 18, 11, 19, 17, 15]
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示问题

def draw_coverage(y_line, y_nac, y_combine):
    # x 轴从 1 到 len(y)
    x_values = list(range(1, len(y_line)+1))

# 绘制三条折线
    plt.plot(x_values, y_line, marker='o', linestyle='-', color='b', label='line')
    plt.plot(x_values, y_nac, marker='s', linestyle='--', color='r', label='nac')
    plt.plot(x_values, y_combine, marker='^', linestyle='-.', color='g', label='combine')

    # 添加标题和轴标签
    plt.title("Coverage Variation Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Coverage")

    # 显示图例
    plt.legend()
    os.makedirs('/home/lzq/result/cov_img/', exist_ok=True)
        # 保存图像，不显示
    plt.savefig(f"/home/lzq/result/cov_img/{len(y_line) - 1}_cov.png", dpi=300, bbox_inches='tight')
    # 关闭图像，避免占用内存
    plt.close()

def draw_error(code_error, model_error):
    # x 轴从 1 到 len(y)
    x_values = list(range(1, len(code_error)+1))

# 绘制三条折线
    plt.plot(x_values, code_error, marker='o', linestyle='-', color='b', label='code_error')
    plt.plot(x_values, model_error, marker='s', linestyle='--', color='r', label='model_error')

    # 添加标题和轴标签
    plt.title("Error Numbers Variation Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Error Number")

    # 显示图例
    plt.legend()
    os.makedirs('/home/lzq/result/error_img/', exist_ok=True)

        # 保存图像，不显示
    plt.savefig(f"/home/lzq/result/error_img/{len(code_error) - 1}_error_num.png", dpi=300, bbox_inches='tight')
    # 关闭图像，避免占用内存
    plt.close()


# if __name__ == '__main__':
#     import json

#     # # 读取 JSON 文件
#     # with open("/home/lzq/result/datas/3.json", "r", encoding="utf-8") as f:
#     #     data = json.load(f)

#     # # 输出 JSON 内容
#     # print(type(data['code_errors']))
#     draw_coverage(y_line1, y_nac1, y_nac2)