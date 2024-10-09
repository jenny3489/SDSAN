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

class HSI:
    '''
    A class for Hyperspectral Image (HSI) data.
    '''
    def __init__(self,data, rows, cols, gt,sgt,patch_size):
        if data.shape[0] < data.shape[1]:
            data = data.transpose()

        self.bands = np.min(data.shape)
        self.rows=rows
        self.cols=cols
        self.p=gt.shape[0]

        ####padding
        image = np.reshape(data, (self.rows, self.cols, self.bands))
        h=rows
        w=cols
        h1=h//patch_size if h//patch_size==0 else h // patch_size + 1
        w1=w//patch_size if w//patch_size==0 else w//patch_size+1
        image_pad = np.pad(image, ((0, patch_size * h1 - h), (0, patch_size * w1 - w), (0, 0)),'edge')
      
        self.prows = image_pad.shape[0]
        self.pcols = image_pad.shape[1]
        self.image_pad = image_pad
        self.image = image

        self.gt = gt
        self.sgt=sgt
    
    def array(self,n):
        """this returns a array of spectra with shape num pixels x num bands
        
        Returns:
            a matrix -- array of spectra
        """
        if n==1:
            return np.reshape(self.image_pad,(self.prows*self.pcols,self.bands))
        else:
            return np.reshape(self.image, (self.rows * self.cols, self.bands))
    
    def get_bands(self, bands):
        return self.image[:,:,bands]

    def crop_image(self,start_x,start_y,delta_x=None,delta_y=None):
        if delta_x is None: delta_x = self.cols - start_x
        if delta_y is None: delta_y = self.rows - start_y
        self.cols = delta_x
        self.rows = delta_y
        self.image = self.image[start_x:delta_x+start_x,start_y:delta_y+start_y,:]
        return self.image


def load_HSI(path,patch_size=4):
    try:
        data = sio.loadmat(path)
    except NotImplementedError:
        data = h5py.File(path, 'r')
    
    numpy_array = np.asarray(data['Y'], dtype=np.float32)
    numpy_array = numpy_array / np.max(numpy_array.flatten())
    n_rows = data['lines'].item()
    n_cols = data['cols'].item()
    
    if 'GT' in data.keys():
        gt = np.asarray(data['GT'], dtype=np.float32)
    else:
        gt = None
    if 'S_GT' in data.keys():
        sgt = np.asarray(data['S_GT'], dtype=np.float32)
    else:
        sgt = None
    return HSI(numpy_array, n_rows, n_cols, gt,sgt,patch_size)

def numpy_SID(y_true, y_pred):
    epsilon = 1e-10  # 小常数以避免被零除
    dot_product = np.sum(y_true * y_pred)
    norm_y_true = np.linalg.norm(y_true)
    norm_y_pred = np.linalg.norm(y_pred)
    sid = -np.log((dot_product + epsilon) / (norm_y_true * norm_y_pred + epsilon))
    return sid

def SAD(y_true, y_pred):
    y_true2 = tf.math.l2_normalize(y_true, axis=-1)
    y_pred2 = tf.math.l2_normalize(y_pred, axis=-1)
    A = tf.keras.backend.mean(y_true2 * y_pred2)
    sad = tf.math.acos(A)
    return sad

def numpy_SAD(y_true, y_pred):
    cos = y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if cos>1.0: cos = 1.0
    return np.arccos(cos)


def estimate_snr(Y, r_m, x):
    [L, N] = Y.shape  # L 带（通道）数，N 像素数
    [p , N] = x.shape  # p 端元数量（尺寸减小）

    P_y = sp.sum(Y ** 2) / float(N)
    P_x = sp.sum(x ** 2) / float(N) + sp.sum(r_m ** 2)
    snr_est = 10 * sp.log10((P_x - p / L * P_y) / (P_y - P_x))

    return snr_est

'''SID'''
def SID_order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    mydict = {}
    sid_mat = np.ones((num_endmembers, num_endmembers))

    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sid_mat[i, j] = numpy_SID(endmembers[i, :], endmembersGT[j, :])

    rows = 0
    while rows < num_endmembers:
        minimum = sid_mat.min()
        index_arr = np.where(sid_mat == minimum)
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        if index[0] in mydict.keys():
            sid_mat[index[0], index[1]] = 100
        elif index[1] in mydict.values():
            sid_mat[index[0], index[1]] = 100
        else:
            mydict[index[0]] = index[1]
            sid_mat[index[0], index[1]] = 100
            rows += 1

    total_sid = 0
    num = 0
    for i in range(num_endmembers):
        if np.var(endmembersGT[mydict[i]]) > 0:
            total_sid += numpy_SID(endmembers[i, :], endmembersGT[mydict[i]])
            num += 1

    return mydict, total_sid / float(num)

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


'''SID'''


def SID_plotEndmembersAndGT(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1

    hat, sid = order_endmembers(endmembersGT, endmembers)  # 计算SID而不是SAD
    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()
    title = "SID: " + format(sid, '.3f') + " radians"  # 更新标题以反映SID
    st = plt.suptitle(title)

    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[hat[i], :], 'r', linewidth=1.0)
        plt.plot(endmembersGT[i, :], 'b', linewidth=1.0)
        plt.ylim((0, 1))
        ax.set_title(format(numpy_SID(endmembers[hat[i], :], endmembersGT[i, :]), '.3f'))  # 计算和显示SID
        ax.get_xaxis().set_visible(False)

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.88)
    plt.draw()
    plt.pause(0.001)

def plotEndmembersAndGT(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1

    hat, sad = order_endmembers(endmembersGT, endmembers)
    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()
    title = "mSAD: " + format(sad, '.3f') + " radians"
    st = plt.suptitle(title)

    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[hat[i], :], 'r', linewidth=1.0)
        plt.plot(endmembersGT[i, :], 'b', linewidth=1.0)
        plt.ylim((0,1))
        ax.set_title(format(numpy_SAD(endmembers[hat[i], :], endmembersGT[i, :]), '.3f'))
        ax.get_xaxis().set_visible(False)

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.88)
    plt.draw()
    plt.pause(0.001)

def plotAbundancesSimple(abundances,name):
    abundances = np.transpose(abundances, axes=[1, 0, 2])
    num_endmembers = abundances.shape[2]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0: n = n + 1
    cmap='jet'
    fig =plt.figure(figsize=[12, 12])
    rect = fig.patch
    rect.set_facecolor('white')
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundances[:, :, i], cmap=cmap)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im.set_clim(0, 1)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    fig.savefig(name+'.png')
    plt.close()



'''SID'''


class SID_PlotWhileTraining(callbacks.Callback):
    def __init__(self, plot_every_n, hsi):
        super(SID_PlotWhileTraining, self).__init__()
        self.plot_every_n = plot_every_n
        self.input = hsi.array(n=1)
        self.cols = hsi.cols
        self.rows = hsi.rows
        self.endmembersGT = hsi.gt
        self.sid_values = []
        self.epochs = []

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_loss = []
        self.sid_values = []

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.num_epochs = epoch
        endmembers = self.model.layers[-1].get_weights()[0]
        endmembers = np.squeeze(endmembers)

        if self.plot_every_n == 0 or epoch % self.plot_every_n != 0:
            return
        if self.endmembersGT is not None:
            plotEndmembersAndGT(self.endmembersGT, endmembers)
        else:
            plotEndmembers(endmembers)
        # Calculate and store SID
        hat, sid = SID_order_endmembers(endmembers, self.endmembersGT)
        self.sid_values.append(sid)
        self.epochs.append(epoch)

class PlotWhileTraining(callbacks.Callback):
    def __init__(self, plot_every_n, hsi):
        super(PlotWhileTraining, self).__init__()
        self.plot_every_n = plot_every_n
        self.input = hsi.array(n=1)
        self.cols = hsi.cols
        self.rows = hsi.rows
        self.endembersGT = hsi.gt
        self.sads = None
        self.epochs = []

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_loss = []
        self.sads = []

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('SAD'))
        self.num_epochs = epoch
        endmembers = self.model.layers[-1].get_weights()[0]
        endmembers = np.squeeze(endmembers)

        if self.plot_every_n == 0 or epoch % self.plot_every_n != 0:
            return
        if self.endmembersGT is not None:
            plotEndmembersAndGT(self.endmembersGT, endmembers)
        else:
            plotEndmembers(endmembers)

def reconstruct(A,S):
    s_shape = S.shape
    S = np.reshape(S,(S.shape[0]*S.shape[1],S.shape[2]))
    reconstructed = np.matmul(S,A)
    reconstructed = np.reshape(reconstructed, (s_shape[0], s_shape[1],reconstructed.shape[1]))
    return reconstructed

def compute_sid(gt, A):
    hat, sid = SID_order_endmembers(gt, A)  # 计算SID而不是SAD
    num_endmembers = A.shape[0]
    sid_mat = [0] * num_endmembers
    for i in range(num_endmembers):
        sid_mat[i] = numpy_SID(A[hat[i], :], gt[i, :])  # 使用SID替代SAD
    return sid_mat, sid

def compute_sad(gt,A):
    hat, sad = order_endmembers(gt,A)
    num_endmembers=A.shape[0]
    sad_mat = [0]*num_endmembers
    for i in range(num_endmembers):
        sad_mat[i]=numpy_SAD(A[hat[i], :], gt[i, :])
    return sad_mat,sad

def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape
    class_rmse = [0] * c
    for i in range(c):
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]) ** 2).sum() / (w * h))
    mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))
    return class_rmse, mean_rmse

def plotEndmembers(endmembers):          #可视化端元
    num_endmembers = endmembers.shape[0]
    endmembers = endmembers / endmembers.max()
    fig = plt.figure(1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, 2, i + 1)
        plt.plot(endmembers[i, :], 'r', linewidth=1.0)
        ax.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

'''rmse'''
def numpy_RMSE(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

def MSE_order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    mydict = {}
    rmse_mat = np.ones((num_endmembers, num_endmembers))

    for i in range(num_endmembers):
        for j in range(num_endmembers):
            rmse_mat[i, j] = numpy_RMSE(endmembers[i, :], endmembersGT[j, :])

    rows = 0
    while rows < num_endmembers:
        minimum = rmse_mat.min()
        index_arr = np.where(rmse_mat == minimum)
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        if index[0] in mydict.keys():
            rmse_mat[index[0], index[1]] = 100
        elif index[1] in mydict.values():
            rmse_mat[index[0], index[1]] = 100
        else:
            mydict[index[0]] = index[1]
            rmse_mat[index[0], index[1]] = 100
            rows += 1

    total_rmse = 0
    num = 0
    for i in range(num_endmembers):
        if np.var(endmembersGT[mydict[i]]) > 0:
            total_rmse += numpy_RMSE(endmembers[i, :], endmembersGT[mydict[i]])
            num += 1

    return mydict, total_rmse / float(num)

def MSE_plotEndmembersAndGT(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1

    hat, rmse = MSE_order_endmembers(endmembersGT, endmembers)
    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()
    title = "RMSE: " + format(rmse, '.3f')
    st = plt.suptitle(title)

    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[hat[i], :], 'r', linewidth=1.0)
        plt.plot(endmembersGT[i, :], 'b', linewidth=1.0)
        plt.ylim((0, 1))
        ax.set_title(format(numpy_RMSE(endmembers[hat[i], :], endmembersGT[i, :]), '.3f'))
        ax.get_xaxis().set_visible(False)

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.88)
    plt.draw()
    plt.pause(0.001)

# Modify the PlotWhileTraining class to collect and plot RMSE values
class MSE_PlotWhileTraining(callbacks.Callback):
    def __init__(self, plot_every_n, hsi):
        super(MSE_PlotWhileTraining, self).__init__()
        self.plot_every_n = plot_every_n
        self.input = hsi.array(n=1)
        self.cols = hsi.cols
        self.rows = hsi.rows
        self.endmembersGT = hsi.gt
        self.rmse_values = []
        self.epochs = []

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_loss = []
        self.rmse_values = []

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.num_epochs = epoch
        endmembers = self.model.layers[-1].get_weights()[0]
        endmembers = np.squeeze(endmembers)

        if self.plot_every_n == 0 or epoch % self.plot_every_n != 0:
            return
        if self.endmembersGT is not None:
            MSE_plotEndmembersAndGT(self.endmembersGT, endmembers)
        else:
            plotEndmembers(endmembers)
        # Calculate and store RMSE
        hat, rmse = MSE_order_endmembers(endmembers, self.endmembersGT)
        self.rmse_values.append(rmse)
        self.epochs.append(epoch)

def vca(Y, R, verbose=True, snr_input=0):
    # 顶点分量分析
    #
    # Ae，指数，Yp = vca(Y,R,verbose = True,snr_input = 0)
    #
    # ------- Input variables -------------
    #  Y - 尺寸为 L（通道）x N（像素）的矩阵
    #      每个像素都是 R 端元的线性混合
    #      signatures Y = M x s, where s = gamma x alfa
    #      gamma is a illumination perturbation factor and
    #      alfa are the abundance fractions of each endmember.
    #  R - positive integer number of endmembers in the scene
    #
    # ------- Output variables -----------
    # Ae     - estimated mixing matrix (endmembers signatures)
    # indice - pixels that were chosen to be the most pure
    # Yp     - Data matrix Y projected.
    #
    # ------- Optional parameters---------
    # snr_input - (float) signal to noise ratio (dB)
    # v         - [True | False]
    # ------------------------------------
    #
    # Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)
    # This code is a translation of a matlab code provided by
    # Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
    # available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
    # Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))
    #
    # more details on:
    # Jose M. P. Nascimento and Jose M. B. Dias
    # "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
    # submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
    #
    #

    #############################################
    # Initializations
    #############################################
    if len(Y.shape) != 2:
        sys.exit('Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')

    [L, N] = Y.shape  # L number of bands (channels), N number of pixels

    R = int(R)
    if (R < 0 or R > L):
        sys.exit('ENDMEMBER parameter must be integer between 1 and L')

    #############################################
    # SNR Estimates
    #############################################

    if snr_input == 0:
        y_m = sp.mean(Y, axis=1, keepdims=True)
        Y_o = Y - y_m  # data with zero-mean
        Ud = splin.svd(sp.dot(Y_o, Y_o.T) / float(N))[0][:, :R]  # computes the R-projection matrix
        x_p = sp.dot(Ud.T, Y_o)  # project the zero-mean data onto p-subspace

        SNR = estimate_snr(Y, y_m, x_p)

        if verbose:
            print("SNR estimated = {}[dB]".format(SNR))
    else:
        SNR = snr_input
        if verbose:
            print("input SNR = {}[dB]\n".format(SNR))

    SNR_th = 15 + 10 * sp.log10(R)

    #############################################
    # Choosing Projective Projection or
    #          projection to p-1 subspace
    #############################################

    if SNR < SNR_th:
        if verbose:
            print("... Select proj. to R-1")

            d = R - 1
            if snr_input == 0:  # it means that the projection is already computed
                Ud = Ud[:, :d]
            else:
                y_m = sp.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m  # data with zero-mean

                Ud = splin.svd(sp.dot(Y_o, Y_o.T) / float(N))[0][:, :d]  # computes the p-projection matrix
                x_p = sp.dot(Ud.T, Y_o)  # project thezeros mean data onto p-subspace

            Yp = sp.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

            x = x_p[:d, :]  # x_p =  Ud.T * Y_o is on a R-dim subspace
            c = sp.amax(sp.sum(x ** 2, axis=0)) ** 0.5
            y = sp.vstack((x, c * sp.ones((1, N))))
    else:
        if verbose:
            print("... Select the projective proj.")

        d = R
        Ud = splin.svd(sp.dot(Y, Y.T) / float(N))[0][:, :d]  # computes the p-projection matrix

        x_p = sp.dot(Ud.T, Y)
        Yp = sp.dot(Ud, x_p[:d, :])  # again in dimension L (note that x_p has no null mean)

        x = sp.dot(Ud.T, Y)
        u = sp.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
        y = x / sp.dot(u.T, x)

    #############################################
    # VCA algorithm
    #############################################

    indice = sp.zeros((R), dtype=int)
    A = sp.zeros((R, R))
    A[-1, 0] = 1

    for i in range(R):
        w = np.random.rand(R, 1);
        f = w - sp.dot(A, sp.dot(splin.pinv(A), w))
        f = f / splin.norm(f)

        v = sp.dot(f.T, y)

        indice[i] = sp.argmax(sp.absolute(v))
        A[:, i] = y[:, indice[i]]  # same as x(:,indice(i))

    Ae = Yp[:, indice]

    return Ae, indice, Yp