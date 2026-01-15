import json
import csv
import matplotlib.pyplot as plt
import numpy as np


def get_cov(cov_name, num):
    exp1_file_path = f'/media/lzq/My Passport/pylot实验数据/实验结果/exp{num}/exp{num}/100.json'
    with open(exp1_file_path, 'r', encoding='utf-8') as json_file:
        data1 = json.load(json_file)

    line_cov_lens=[]
    nac_lens=[]

    lines_map=None
    nac_vec=None

    max_com_cov=0
    # print(data1.keys())
    for i in range(100):
        if data1[cov_name][i] > max_com_cov or \
            (cov_name == 'plt_pnac_cov' and len(data1['pnac_map'][i]) > len(data1['pnac_map'][i-1])) or \
            ((cov_name=='plt_combine_cov' or cov_name == 'plt_line_cov') and count_line_map_1(data1['line_status_vector_maps'][i]) > count_line_map_1(data1['line_status_vector_maps'][i-1])):
            if data1[cov_name][i] > max_com_cov:
                max_com_cov = data1[cov_name][i]

            if lines_map is None:
                lines_map = data1['cur_line_status_vector_maps'][i]
                print(type(lines_map))
            else:
                lines_map= merge_line_map(data1['cur_line_status_vector_maps'][i], lines_map)
            # print(type(lines_map))
            count_1, count_01 = count_line_map_01(lines_map)
            line_cov_lens.append(count_1)

            if nac_vec is None:
                nac_vec = data1['seed_vectors'][i]['nac_vector']
            else:
                nac_vec = merge_nac_vec(nac_vec, data1['seed_vectors'][i]['nac_vector'])
            nac_lens.append(sum(nac_vec))


        else:
            line_cov_lens.append(line_cov_lens[-1])
            nac_lens.append(nac_lens[-1])
    
    nac_vac_len = len(nac_vec)
    _, line_vec_len = count_line_map_01(lines_map)

    line_cov = [x / 3209 for x in line_cov_lens]
    nac_cov = [x / 4296 for x in nac_lens]

    return line_cov, nac_cov, line_vec_len, nac_vac_len


def merge_nac_vec(A, B):
    merged=[]
    for a, b in zip(A, B):
        merged.append(max(a, b))
    return merged
# def count_nac_map_01(merged):
#     return sum(merged), len(merged)


def merge_line_map(A, B):
    all_keys = set(A.keys()) | set(B.keys())
    merged={}
    for key in all_keys:
        if key not in A:
            merged[key] = B[key]
        elif key not in B:
            merged[key] = A[key]
        else:
            merged[key]=[]
            for a, b in zip(A[key], B[key]):
                if a==-1 or b==-1:
                    merged[key].append(-1)
                else:
                    merged[key].append(max(a, b))
    
    return merged

def count_line_map_01(merged):
    count_1 = 0
    count_01 = 0
    for lst in merged.values():
        for val in lst:
            if val == 1:
                count_1 += 1
                count_01 += 1
            elif val == 0:
                count_01 += 1
    return count_1, count_01

def count_line_map_1(merged):
    count_1 = 0
    count_01 = 0
    for lst in merged.values():
        for val in lst:
            if val == 1:
                count_1 += 1
                count_01 += 1
            elif val == 0:
                count_01 += 1
    return count_1


def draw_coverage(title, y1, y2, y3, y4, path):
    # x 轴从 1 到 len(y)

    x = np.arange(0, len(y1))
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
    plt.ylabel("Coverage(%)")
    # plt.ylim(0.2, 0.7)

    # 显示图例
    plt.legend(loc='best',framealpha=0.5)
    plt.grid(alpha=0.5) 

        # 保存图像，不显示
    plt.savefig(path, dpi=300, bbox_inches='tight')
    # 关闭图像，避免占用内存
    plt.close()


if __name__ == '__main__':
    deephunter_line_cov_1, deephunter_nac_cov_1, line_vec_len, nac_vac_len = get_cov('plt_nac_cov', 7)
    pythonfuzz_line_cov_1, pythonfuzz_nac_cov_1, line_vec_len, nac_vac_len = get_cov('plt_line_cov', 10)
    combine_line_cov_1, combine_nac_cov_1, line_vec_len, nac_vac_len = get_cov('plt_combine_cov', 1)
    pnac_line_cov_1, pnac_nac_cov_1, line_vec_len, nac_vac_len = get_cov('plt_pnac_cov', 4)


    deephunter_line_cov_3, deephunter_nac_cov_3, line_vec_len, nac_vac_len = get_cov('plt_nac_cov', 21)
    pythonfuzz_line_cov_3, pythonfuzz_nac_cov_3, line_vec_len, nac_vac_len = get_cov('plt_line_cov', 24)
    combine_line_cov_3, combine_nac_cov_3, line_vec_len, nac_vac_len = get_cov('plt_combine_cov', 15)
    pnac_line_cov_3, pnac_nac_cov_3, line_vec_len, nac_vac_len = get_cov('plt_pnac_cov', 18)

    deephunter_line_cov_1 =[0]+ [x * 100 for x in deephunter_line_cov_1]
    deephunter_nac_cov_1 = [0]+[x * 100 for x in deephunter_nac_cov_1]
    pythonfuzz_line_cov_1 = [0]+[x * 100 for x in pythonfuzz_line_cov_1]
    pythonfuzz_nac_cov_1 =[0]+ [x * 100 for x in pythonfuzz_nac_cov_1]
    combine_line_cov_1 = [0]+[x * 100 for x in combine_line_cov_1]
    combine_nac_cov_1 = [0]+[x * 100 for x in combine_nac_cov_1]
    pnac_line_cov_1 = [0]+[x * 100 for x in pnac_line_cov_1]
    pnac_nac_cov_1 = [0]+[x * 100 for x in pnac_nac_cov_1]

    deephunter_line_cov_3 =[0]+[x * 100 for x in deephunter_line_cov_3]
    deephunter_nac_cov_3 =[0]+[x * 100 for x in deephunter_nac_cov_3]
    pythonfuzz_line_cov_3 = [0]+[x * 100 for x in pythonfuzz_line_cov_3]
    pythonfuzz_nac_cov_3 = [0]+[x * 100 for x in pythonfuzz_nac_cov_3]
    combine_line_cov_3 =[0]+ [x * 100 for x in combine_line_cov_3]
    combine_nac_cov_3 =[0]+[x * 100 for x in combine_nac_cov_3]
    pnac_line_cov_3 = [0]+[x * 100 for x in pnac_line_cov_3]
    pnac_nac_cov_3 = [0]+[x * 100 for x in pnac_nac_cov_3]

    draw_coverage("Obstacle Detection Code Coverage",deephunter_line_cov_1, pythonfuzz_line_cov_1, combine_line_cov_1, pnac_line_cov_1, "/home/lzq/test_reslut/会议论文实验结果/覆盖图/ob_town1_line_from0.png")
    draw_coverage("Obstacle Detection Neuron Coverage",deephunter_nac_cov_1, pythonfuzz_nac_cov_1, combine_nac_cov_1, pnac_nac_cov_1, "/home/lzq/test_reslut/会议论文实验结果/覆盖图/ob_town1_nac_from0.png")

    draw_coverage("Traffic Light Detection Code Coverage", deephunter_line_cov_3, pythonfuzz_line_cov_3, combine_line_cov_3, pnac_line_cov_3, "/home/lzq/test_reslut/会议论文实验结果/覆盖图/tl_town3_line_from0.png")
    draw_coverage("Traffic Light Detection Neuron Coverage", deephunter_nac_cov_3, pythonfuzz_nac_cov_3, combine_nac_cov_3, pnac_nac_cov_3, "/home/lzq/test_reslut/会议论文实验结果/覆盖图/tl_town3_nac_from0.png")
# if __name__ == '__main__':
#     line_cov, nac_cov, line_vec_len, nac_vac_len = get_cov('plt_line_cov', 23)
#     print(f'line_cov:{line_cov[-1]}')
#     print(f'nac_cov:{nac_cov[-1]}')
#     print(line_vec_len, nac_vac_len)

# if __name__ == '__main__':
#     covs = ['plt_combine_cov', 'plt_combine_cov', 'plt_combine_cov', 
#             'plt_pnac_cov', 'plt_pnac_cov', 'plt_pnac_cov',
#             'plt_nac_cov', 'plt_nac_cov', 'plt_nac_cov',
#             'plt_line_cov', 'plt_line_cov', 'plt_line_cov',
#             'plt_combine_cov', 'plt_combine_cov', 'plt_combine_cov', 
#             'plt_pnac_cov', 'plt_pnac_cov', 'plt_pnac_cov',
#             'plt_nac_cov', 'plt_nac_cov', 'plt_nac_cov',
#             'plt_line_cov', 'plt_line_cov', 'plt_line_cov']
#     maps=[]
#     finall_map = {}
#     for i in range(24):
#         finall_map= merge_line_map(get_cov(covs[i], i+1), finall_map)
#     _, line_vec_len = count_line_map_01(finall_map)
#     print(line_vec_len)
    
        
    
