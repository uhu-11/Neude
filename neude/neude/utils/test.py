import numpy as np
import matplotlib.pyplot as plt

# 假设 data 和 img_ground_truth 是两个轨迹的坐标点
data = [[0, 0], [1, 2], [2, 1], [3, 3]]  # 示例轨迹 1
img_ground_truth = [[0, 1], [1, 3], [2, 2], [3, 4]]  # 示例轨迹 2

# 将轨迹转换为 NumPy 数组
data_array = np.array(data)
ground_truth_array = np.array(img_ground_truth)

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制轨迹 1
plt.plot(data_array[:, 0], data_array[:, 1], marker='o', linestyle='-', color='b', label='Trajectory 1')

# 绘制轨迹 2
plt.plot(ground_truth_array[:, 0], ground_truth_array[:, 1], marker='o', linestyle='--', color='r', label='Trajectory 2')

# 添加标题和标签
plt.title('Trajectory Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 添加图例
plt.legend()

# 显示网格
plt.grid()

# 保存图形到指定路径
save_path = '/path/to/save/trajectory_plot.png'  # 替换为你想保存的路径
plt.show()
