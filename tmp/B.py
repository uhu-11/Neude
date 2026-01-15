# import pickle
# with open( '/home/lzq/experiment_datatset/inputs/scene_datas/town4_obstacle/scene_data_rs.pkl', "rb") as f:
#     data=pickle.load(f)
# print(data)

import json
k = {i:[1,2]*1000 for i in range(1,5000)}
k2 = {i:[1,2]*1000 for i in range(1,5000)}
k3 = {1:[1,2]*1000 for i in range(1,5000)}

a=json.dumps(k, sort_keys= True)
a2=json.dumps(k2, sort_keys= True)
a3=json.dumps(k3, sort_keys= True)
c={a:0.1, a3:1}

print(a2 in c.keys())