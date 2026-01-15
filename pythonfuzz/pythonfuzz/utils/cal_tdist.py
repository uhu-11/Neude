import traj_dist.distance as tdist
import os
import numpy as np
import glob
def get_set_tdist(predictions_path, all_imgs_ground_truths, iter, target_planning_type_enum):
    npy_files = glob.glob(os.path.join(predictions_path, "planning_*.npy"))
    npy_files.sort()
    imgs_dist = []
    sspd_dist=[]
    # print('all_imgs_ground_truths', len(all_imgs_ground_truths))
    # data = np.load(npy_files[i], allow_pickle=True)
    # iht=all_imgs_ground_truths[i]
    # tdist.lcss(data, np.array(iht), eps=0.05)/float(min(len(data), len(iht)))
    # all_imgs_ground_truths = np.load(ground_truth_path, allow_pickle=True)  #所有图像的ground_truth
    os.makedirs('/media/lzq/D/lzq/pylot_test/pythonfuzz/tdists', exist_ok=True)
    for i, img_ground_truth in enumerate(all_imgs_ground_truths):    #一个图像的ground——truth，是一个list，里面包含轨迹信息
        data = np.load(npy_files[i], allow_pickle=True)
        # print('planning_data_size:', len(data))
        # print('planning_ground_size:', len(np.array(img_ground_truth)))
        # print('tdist.lcss(data, np.array(img_ground_truth))',tdist.lcss(data, np.array(img_ground_truth)))
        imgs_dist.append(tdist.lcss(data, np.array(img_ground_truth), eps=1))
        # path = '/media/lzq/D/lzq/pylot_test/pythonfuzz/tdists/tdist_'+str(iter)+'_'+str(i)+'_'+str(target_planning_type_enum)+'.png'
        # draw_tdist(data, np.array(img_ground_truth), 'tdist_'+str(iter)+'_'+str(i)+'_'+str(target_planning_type_enum), path)
        # sspd_dist.append(tdist.sspd(data, np.array(img_ground_truth)))
    # iter_iou = np.mean(imgs_iou)
    
    print('imgs_dist',imgs_dist)
    # print('sspd_dist', sspd_dist)
    return imgs_dist

def get_set_steer_diff(predictions_path, all_imgs_ground_truths, iter, target_planning_type_enum):
    npy_files = glob.glob(os.path.join(predictions_path, "control_*.npy"))
    npy_files.sort()
    steers = []
    ground_truth_steers = []
    imgs_steer_diff = []
    print('all_imgs_ground_truths', len(all_imgs_ground_truths))
    # all_imgs_ground_truths = np.load(ground_truth_path, allow_pickle=True)  #所有图像的ground_truth
    for i, img_ground_truth in enumerate(all_imgs_ground_truths):    #一个图像的ground——truth，是一个list，里面包含轨迹信息
        data = np.load(npy_files[i], allow_pickle=True)
        # print('img_ground_truth!!!!!!!:',img_ground_truth)
        imgs_steer_diff.append(abs(img_ground_truth['steer']-data.item()['steer']))
        steers.append(data.item()['steer'])
        ground_truth_steers.append(img_ground_truth['steer'])

    path = '/media/lzq/D/lzq/pylot_test/pythonfuzz/steer/steer_'+str(iter)+'_'+str(target_planning_type_enum)+'.png'
    os.makedirs('/media/lzq/D/lzq/pylot_test/pythonfuzz/steer', exist_ok=True)
    draw_steer_diff(ground_truth_steers, steers, 'steer_'+str(iter)+'_'+str(target_planning_type_enum), path)
    print('imgs_steer_diff',imgs_steer_diff)
    return imgs_steer_diff




# # 指定文件路径
# file_path = '/home/lzq/experiment_datatset/fuzz_test_dataset/town1/obstacles_y.npy'

# # 读取 .npy 文件
# try:
#     data = np.load(file_path, allow_pickle=True)  # allow_pickle=True 允许加载包含 Python 对象的数组
#     print("文件内容:")
#     print(data)  # 打印读取的数据
# except Exception as e:
#     print(f"读取文件时出错: {e}")
import numpy as np
import matplotlib.pyplot as plt
def draw_tdist(data_array,ground_truth_array, name, path):
    plt.figure(figsize=(8, 6))

    # 绘制轨迹 1
    plt.plot(data_array[:, 0], data_array[:, 1], marker='o', linestyle='-', color='b', label='Trajectory 1')

    # 绘制轨迹 2
    plt.plot(ground_truth_array[:, 0], ground_truth_array[:, 1], marker='o', linestyle='--', color='r', label='ground_truth')

    # 添加标题和标签
    plt.title(name)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid()

    plt.savefig(path, dpi=300, bbox_inches='tight')  # 保存图形

def draw_steer_diff(steering_angles_1,steering_angles_2, name, path):
    # 将角度转换为弧度

    angles_rad_1 = np.radians(steering_angles_1)
    angles_rad_2 = np.radians(steering_angles_2)

    # 确保两个列表的起始点一致
    # 这里假设我们希望两个列表的起始点都为 0
    if steering_angles_1[0] != steering_angles_2[0]:
        # 在较短的列表前面添加零值
        if len(steering_angles_1) < len(steering_angles_2):
            steering_angles_1 = [0] + steering_angles_1
        else:
            steering_angles_2 = [0] + steering_angles_2

    # 重新计算角度
    angles_rad_1 = np.radians(steering_angles_1)
    angles_rad_2 = np.radians(steering_angles_2)

    # 创建图形
    plt.figure(figsize=(10, 10))

    # 初始化起点
    start_x_1, start_y_1 = 0, 0
    start_x_2, start_y_2 = 0, 0

    # 存储路径点
    path_x_1, path_y_1 = [start_x_1], [start_y_1]
    path_x_2, path_y_2 = [start_x_2], [start_y_2]

    # 绘制第一个转向角的路径
    for angle in angles_rad_1:
        x = np.sin(angle)  # X 坐标
        y = np.cos(angle)  # Y 坐标
        start_x_1 += x
        start_y_1 += y
        path_x_1.append(start_x_1)
        path_y_1.append(start_y_1)

    # 绘制第二个转向角的路径
    for angle in angles_rad_2:
        x = np.sin(angle)  # X 坐标
        y = np.cos(angle)  # Y 坐标
        start_x_2 += x
        start_y_2 += y
        path_x_2.append(start_x_2)
        path_y_2.append(start_y_2)

    # 绘制路径
    plt.plot(path_x_1, path_y_1, marker='o', linestyle='-', color='b', label='ground_truth')
    plt.plot(path_x_2, path_y_2, marker='o', linestyle='--', color='r', label='Trajectory')

    # 设置坐标轴范围
    plt.xlim(-5, 5)

    # 添加标题和标签
    plt.title(name)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 添加图例
    plt.legend()

    # 添加网格
    plt.grid()

    # 添加坐标轴比例
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(path, dpi=300, bbox_inches='tight')  # 保存图形