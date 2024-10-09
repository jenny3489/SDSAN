# 作者:王勇
# 开发时间:2023/11/7 17:59
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
import scipy as sp

def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape
    class_rmse = [0] * c
    for i in range(c):
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]) ** 2).sum() / (w * h))
    mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))
    return class_rmse , mean_rmse

def plotAbundancesSimple(abundances):
    num_endmembers = abundances.shape[2]
    cmap='jet'
    if not os.path.exists("./sy_abandance"):
        os.makedirs("./sy_abandance")


    for i in range(num_endmembers):
        fig = plt.figure(figsize=[6, 6])
        rect = fig.patch
        rect.set_facecolor('white')

        ax = plt.subplot(1, 1, 1)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundances[:, :, i], cmap=cmap)
        # plt.colorbar(im, orientation='vertical',pad=0.02,ax = ax,shrink=0.8)
        # cax = plt.colorbar(im, orientation='vertical', pad=0.02, ax=ax,shrink=0.8)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im.set_clim(0, 1)

        plt.tight_layout()
        save_path = os.path.join("./sy_abandance", f'{"GTabuandance"}{i}.svg')  # 使用f-string格式化文件名
        plt.savefig(save_path, format='svg')  # 保存为SVG格式
        plt.show()
        # plt.savefig(f"{'./result_p/VCA'}_endmember_{i + 1}.png")
        plt.close()

data = sio.loadmat("./Results/DAEU_SSABN/sy_data/sy_data_run1.mat")
data_t = sio.loadmat("./Datasets/sy20.mat")

A = data["S"]   #main S,DH A_est, TA A
B = data_t['S_GT']
A= np.transpose(A, (1, 0, 2))  #if DH
A = A[:,:,(4,2,3,0,1)]
plotAbundancesSimple(A)
plotAbundancesSimple(B)
msre,mean = compute_rmse(B,A)
for i in range(5):
    print("Class", i + 1, ":", msre[i])
print(mean)
