# 作者:王勇
# 开发时间:2024/1/27 18:44
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
import h5py
import scipy as sp
def plotEndmembersAndGT( endmembersGT,wavelength_range=(400,2500)):
    num_endmembers,num_bands = endmembersGT.shape
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1
    wavelengths=np.linspace(wavelength_range[0],wavelength_range[1],num_bands)

    plt.plot(wavelengths, endmembersGT[0], 'r', linewidth=2.0)
    plt.plot(wavelengths,endmembersGT[1], 'b', linewidth=2.0)
    plt.plot(wavelengths, endmembersGT[2], 'g', linewidth=2.0)
    plt.ylim((0.2,1))
    plt.yticks(np.arange(0,1.1,0.2))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Wavelength (nm)',fontsize=15)
    plt.ylabel('Reflectance',fontsize=15)
    plt.grid(True)
    plt.draw()
    plt.pause(0.001)
data_t = sio.loadmat("./Datasets/Samson.mat")
A = data_t['GT']
plotEndmembersAndGT(A)