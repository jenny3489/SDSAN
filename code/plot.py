# 作者:王勇
# 开发时间:2023/11/6 17:08
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras import callbacks
from scipy import io as sio
import matplotlib.pyplot as plt
import sys
import warnings
import scipy.linalg as splin
warnings.filterwarnings("ignore")
import numpy as np
import os
import h5py
import scipy as sp

def numpy_SAD(y_true, y_pred):
    cos = y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if cos>1.0: cos = 1.0
    return np.arccos(cos)

def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    mydict = {}
    sad_mat = np.ones((num_endmembers, num_endmembers))
    #for i in range(num_endmembers):
        #endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        #endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_SAD(endmembers[i, :], endmembersGT[j, :])
    rows = 0
    while rows < num_endmembers:
        minimum = sad_mat.min()
        index_arr = np.where(sad_mat == minimum)
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        if index[0] in mydict.keys():
            sad_mat[index[0], index[1]] = 100
        elif index[1] in mydict.values():
            sad_mat[index[0], index[1]] = 100
        else:
            mydict[index[0]] = index[1]
            sad_mat[index[0], index[1]] = 100
            rows += 1
    ASAM = 0
    num = 0
    for i in range(num_endmembers):
        if np.var(endmembersGT[mydict[i]]) > 0:
            ASAM = ASAM + numpy_SAD(endmembers[i, :], endmembersGT[mydict[i]])
            num += 1

    return mydict, ASAM / float(num)

def plotEndmembersAndGT(endmembers, endmembersGT,wavelength_range=(0,200)):
    if not os.path.exists("./urban_endmembers"):
        os.makedirs("./urban_endmembers")
    num_endmembers,num_bands = endmembers.shape
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1

    hat, sad = order_endmembers(endmembersGT, endmembers)

    wavelengths=np.linspace(wavelength_range[0],wavelength_range[1],num_bands)

    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    for i in range(num_endmembers):
        plt.plot(wavelengths,endmembers[hat[i], :], 'r', linewidth=2.0)
        plt.plot(wavelengths,endmembersGT[i, :], 'b', linewidth=2.0)
        plt.ylim((0.2,1))
        plt.yticks(np.arange(0,1.1,0.2))
        plt.xlabel('Band', fontsize=15)
        # plt.xlabel('Wavelength (nm)',fontsize=15)
        plt.ylabel('Reflectance',fontsize=15)
        save_path = os.path.join("./urban_endmembers", f'{"DHendmembers"}{i}.svg') # 使用f-string格式化文件名
        plt.savefig(save_path, format='svg')  # 保存为SVG格式
        plt.show()
        plt.close()

data = sio.loadmat("./Results/CNN_SSABN/sy20_data/sy20_data_run1.mat")
data_t = sio.loadmat("./Datasets/sy20.mat")
A = data['A']     #main A,DH E_est,TA E
print(A.shape)
B = data_t['GT']
# A= np.transpose(A, (1, 0))    #if DH,运行
plotEndmembersAndGT(A,B)


