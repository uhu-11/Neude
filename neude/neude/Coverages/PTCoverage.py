import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'neude'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'PTtool'))
from PTtool.pt.TriangularProjection import TriProCover
class PTCoverage():
    def get_coverage(self, x_test_prob_matrix, y, nb_classes, deep_num):
        #print("x_test_prob_matrix:", x_test_prob_matrix)
        #print("y:", y)
     
        tripro_cover = TriProCover()
        sp_c_arr, sp_v_arr,cov_num_arr_list,no_cov_num_arr_list = tripro_cover.cal_triangle_cov(x_test_prob_matrix, y, nb_classes, deep_num,
                                                        by_deep_num=True)
        #print("sp_c_arr:", sp_c_arr)
        pt_rate = sp_c_arr[len(sp_c_arr)-1]
        #print("pt_rate:", pt_rate)
        region_length = tripro_cover.get_total_bins_num(deep_num, len(sp_c_arr))
        pt_vector = tripro_cover.cal_cov_bins(x_test_prob_matrix, y, nb_classes, deep_num)
        return pt_rate, region_length, pt_vector

if __name__ == "__main__":
    x_test_prob_matrix = np.array([[3.3277999e-07, 1.2358712e-08, 9.9999964e-01, 9.8931377e-11, 2.2347471e-08, 7.0623001e-11, 5.7113540e-11, 1.5605305e-08, 1.9235001e-10, 3.4074379e-09], 
                                   [3.0076171e-06,9.9999678e-01, 4.9576581e-08, 5.7510002e-10, 3.7022517e-08, 2.6979391e-10, 1.8013399e-10, 2.9772723e-08, 8.3347312e-10, 4.0984570e-08]])
    y = np.array([1, 1])
    x_test_prob_matrix2 = np.array([[3.3277999e-07, 1.2358712e-08, 9.9999964e-01, 9.8931377e-11, 2.2347471e-08, 7.0623001e-11, 5.7113540e-11, 1.5605305e-08, 1.9235001e-10, 3.4074379e-09]])
    y2 = np.array([8])
    x_test_prob_matrix3 = np.array([[3.3277999e-07, 1.2358712e-08, 9.9999964e-01, 9.8931377e-11, 2.2347471e-08, 7.0623001e-11, 5.7113540e-11, 1.5605305e-08, 1.9235001e-10, 3.4074379e-09], 
                                    [9.9999988e-01, 6.8550299e-10, 1.4243085e-07, 1.5076269e-09, 1.5157101e-11, 5.7161542e-10, 1.9265031e-11, 1.9337379e-10, 3.9355834e-09, 5.5645780e-11]])
    y3 = np.array([2, 2])
    nb_classes = 10
    deep_num = 4
    ptCoverage = PTCoverage()
    pt_rate, region_length = ptCoverage.get_coverage(x_test_prob_matrix, y, nb_classes, deep_num)
    # print("pt_rate:", pt_rate)
    # print("region_length:", region_length)
    # pt_rate2, region_length2 = ptCoverage.get_coverage(x_test_prob_matrix2, y2, nb_classes, deep_num)
    # print("pt_rate2:", pt_rate2)
    # print("region_length2:", region_length2)
    # pt_rate3, region_length3 = ptCoverage.get_coverage(x_test_prob_matrix3, y3, nb_classes, deep_num)
    # print("pt_rate3:", pt_rate3)
    # print("region_length3:", region_length3)

    x = np.array([[1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]])
    y = np.array([0])
    x1 = np.array([[1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                   [0.90000000e+00, 0.10000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]])
    y1 = np.array([1, 1])
    pt_rate4, region_length4 = ptCoverage.get_coverage(x, y, nb_classes, deep_num)
    pt_rate5, region_length5 = ptCoverage.get_coverage(x1, y1, nb_classes, deep_num)
    print("pt_rate4:", pt_rate4)
    print("pt_rate5:", pt_rate5)
    # print("pt_rate4:", pt_rate4)
    # print("region_length4:", region_length4)