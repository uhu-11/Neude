import numpy as np
import matplotlib.pyplot as plt
def draw_tdist(data1,data2,name,path):
    plt.figure(figsize=(8, 6))

    # 绘制轨迹 1
    plt.plot(data1[:, 0], data1[:, 1], marker='o', linestyle='-', color='b', label='label')

    # 绘制轨迹 2
    plt.plot(data2[:, 0], data2[:, 1], marker='o', linestyle='--', color='r', label='replay')
    # plt.plot(data3[:, 0], data3[:, 1], marker='v', linestyle='--', color='g', label='rs_samepoints_replay3')
    # plt.plot(data4[:, 0], data4[:, 1], marker='*', linestyle='-', color='k', label='mutate')
    # plt.plot(data5[:, 0], data5[:, 1], marker='x', linestyle=':', color='y', label='mutate2')

    # 添加标题和标签
    plt.title(name)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid()

    plt.savefig(path, dpi=200, bbox_inches='tight')  # 保存图形
    # plt.show()
    plt.close()


def draw_one_tdist(data_array, name, path):
    plt.figure(figsize=(8, 6))

    # 绘制轨迹 1
    plt.plot(data_array[:, 0], data_array[:, 1], marker='o', linestyle='-', color='b', label=name)

   

    # 添加标题和标签
    plt.title(name)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid()

    plt.savefig(path, dpi=200, bbox_inches='tight')  # 保存图形
    # plt.show()
    plt.close()



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
    plt.plot(path_x_1, path_y_1, marker='o', linestyle='-', color='b', label='replay')
    plt.plot(path_x_2, path_y_2, marker='o', linestyle='--', color='r', label='label')

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
    
def draw_plan():
    data1 = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town4/plannings/town4_traffic_light/planning_rs_label_y.npy', allow_pickle=True)
    data2 = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town4/plannings/town4_traffic_light/planning_replay_rs_label_y.npy', allow_pickle=True)
    # data3 = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town1/plannings/exp1/planning_rs_samepoints_replay3.npy', allow_pickle=True)
    # data3 = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town1/plannings/exp1/planning_rs4_replay4.npy', allow_pickle=True)
    # data4 = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town1/plannings/exp1/planning_rs_mutate2_inputSameasMutate1.npy', allow_pickle=True)
    # data5 = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town1/plannings/exp1/planning_rs_mutate.npy', allow_pickle=True)
    # data2 = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town1/plannings/exp1/planning_rs_inputSameasMutate1.npy', allow_pickle=True)
    # print(data1[0])
    for i in range(len(data1)):
        path = f'/home/lzq/experiment_datatset/fuzz_test_dataset/town4/plannings/rsandreplay_imgs/{i:08d}.png'
        draw_tdist(np.array(data1[i]),np.array(data2[i]),'rs',path)
def draw_control():
    data1 = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town1/controls/exp1/control_label_rs_replay_mutation1.npy', allow_pickle=True)
    data2 = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town1/controls/exp1/control_rs_label_y.npy', allow_pickle=True)
    print(type(data1[0]['steer']))
    steers1, steers2 = [],[]
    for i in range(len(data1)):
        steers1.append(data1[i]['steer'])
        steers2.append(data2[i]['steer'])
    print(steers1)

    print('---------------------------------')
    print(steers2)

    path = f'/home/lzq/experiment_datatset/fuzz_test_dataset/town1/controls/exp1/rs_ha_replay.png'
    draw_steer_diff(steers1,steers2,'rs',path)

if __name__=='__main__':
    # data1 = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town1/planningwp_label_y.npy', allow_pickle=True)
    # # draw_tdist(np.array(data[60]), np.array(data[50]),'pic')
    # data2 = np.load('/media/lzq/D/lzq/pylot_test/pylot/predictions/planning/planning_00000050.npy', allow_pickle=True)
    # draw_tdist(np.array(data2),np.array(data1[50]),'pic')
    draw_plan()

    # data1 = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town1/plannings/exp1/planning_wp_label_y.npy', allow_pickle=True)
    # data2 = np.load('/home/lzq/experiment_datatset/fuzz_test_dataset/town1/plannings/exp1/planning_wp_mutate2.npy', allow_pickle=True)
    # # print(data1[0])
    # # for i in range(len(data1[0])):
    # #     # path = f'/home/lzq/experiment_datatset/fuzz_test_dataset/town1/plannings/exp1/wp_mutate_imgs/{i:08d}.png'
    # #     # draw_tdist(np.array(data2[i]),np.array(data1[i]),'wp',path)
    # #     print(data1[0][i], data2[0][i])
    # #     print('-----------------------------------------------')

    # # a=[[1,2],[3,4],[56,7]]
    # # b=[[1,1],[1,1],[1,1]]
    # # print(a-b)
    # print(type(data1[0]))