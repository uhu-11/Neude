import pickle
import time

from numpy import array
from PIL import Image
import os
import numpy as np

from pythonfuzz.config import IMAGE_SAVE_PATH, SEED_SAVE_PATH, PREDICT_ERROR_SEED_PATH, LOCAL_SEED_POOL

''''
    保存种子中的图片到本地路径
'''
def saveImgToPath(seed):
    for ele in seed.params.values():
        if isinstance(ele, Image.Image):
            img = ele
            break
    # img = array(ele)
    # print(img.shape)
    # img = Image.fromarray(np.uint8(img))
    name = str(int(time.time()))+".png"
    # print(IMAGE_SAVE_PATH+"/"+name)
    if not os.path.exists(IMAGE_SAVE_PATH):
        os.makedirs(IMAGE_SAVE_PATH)

    img.save(IMAGE_SAVE_PATH + "/" + name)

##将触发新覆盖的种子放入到指定的文件中
def saveSeedToPickle(seed, name, folder):
    file_pickle_name = name + ".pickle"
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = f'{folder}/{file_pickle_name}'
    with open(path, "wb") as file:
        pickle.dump(seed, file)
    return path

# ##将触发新覆盖的种子放入到指定的文件中
# def saveErrorSeedToPickle(seed, seed_ind):
#     file_pickle_name = str(seed_ind) + ".pickle"
#     if not os.path.exists(LOCAL_SEED_POOL):
#         os.makedirs(file_pickle_name)
#     path = f'{LOCAL_SEED_POOL}/{file_pickle_name}'
#     with open(path, "wb") as file:
#         pickle.dump(seed, file)

def save(seed, excutions, path):
    img = None
    curtime = str(int(time.time()))
    pickle_name = curtime + "_" + str(excutions)
    img_file_pickle_name = path + "/" + pickle_name
    if not os.path.exists(img_file_pickle_name):
        os.makedirs(img_file_pickle_name)
    pickle_path = img_file_pickle_name + "/" + pickle_name + ".pickle"
    with open(pickle_path, "wb") as file:
        pickle.dump(seed, file)
    img_path = img_file_pickle_name +"/" + pickle_name + ".png"

    for ele in seed.params.values():
        if isinstance(ele, Image.Image):
            img = ele
            break
    if img is not None:
        img.save(img_path)
    
    return img_file_pickle_name


def save_error_seed(iter,path,seed):
    os.makedirs(path, exist_ok=True)
    folder_name = iter + '_' + str(int(time.time()))
    os.makedirs(path+"/"+folder_name, exist_ok=True)
    pickle_name = path+"/"+folder_name+"/"+folder_name+".pickle"
    img_name = path+"/"+folder_name+"/"+folder_name+".png"
    with open(pickle_name, "wb") as file:
        pickle.dump(seed, file)
    img = None
    for ele in seed.params.values():
        if isinstance(ele, Image.Image):
            img = ele
            break
    if img is not None:
        if img.size[0] < 50 or img.size[1] < 50:
                img = img.resize((256, 256), Image.LANCZOS) 
        img.save(img_name)
    return path+"/"+folder_name





# 保存种子队列时，每个变异都有一个新文件夹
# 打印总的行以及行覆盖率
# deephunter的图像变异复用