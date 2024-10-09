# 作者:王勇
# 开发时间:2023/10/23 18:00
import random
import time
import tensorflow as tf
from keras import initializers, constraints, layers, activations, regularizers
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import tensor_shape
from utils import HSI, plotEndmembers,SAD,load_HSI,MSE_plotEndmembersAndGT,MSE_PlotWhileTraining,compute_rmse
from utils import plotEndmembersAndGT, plotAbundancesSimple, PlotWhileTraining,SID_plotEndmembersAndGT,SID_PlotWhileTraining
from sklearn.feature_extraction.image import extract_patches_2d
from scipy import io as sio
import os
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def array(x):
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        x= sess.run(x)
    return x

class SumToOne(layers.Layer):
    def __init__(self, **kwargs):
        super(SumToOne, self).__init__(**kwargs)

    def call(self, x):
        x *= K.cast(x >= K.epsilon(), K.floatx())
        x = K.relu(x)
        x = x / (K.sum(x, axis=-1, keepdims=True) + K.epsilon())
        return x


class SparseReLU(tf.keras.layers.Layer):
    def __init__(self,params):
        self.params=params
        super(SparseReLU, self).__init__()
        self.alpha = self.add_weight(shape=(self.params['num_endmembers'],),initializer=tf.keras.initializers.Zeros(),
        trainable=True, constraint=tf.keras.constraints.non_neg())
    def build(self, input_shape):
        self.alpha = self.add_weight(shape=input_shape[1:],initializer=tf.keras.initializers.Zeros(),
        trainable=True, constraint=tf.keras.constraints.non_neg())
        super(SparseReLU, self).build(input_shape)
    def call(self, x):
        return tf.keras.backend.relu(x - self.alpha)


class Autoencoder(object):
    def __init__(self, params):
        self.data = None
        self.params = params
        self.is_deep = True
        self.model = self.create_model()
        self.model.compile(optimizer=self.params["optimizer"], loss=self.params["loss"])

    def create_model(self):
        use_bias = False
        n_end = self.params['num_endmembers']
        # Input layer
        Sparse_ReLU = SparseReLU(self.params)
        input_ = layers.Input(shape=(self.params['n_bands'],))


        if self.is_deep:
            encoded = layers.Dense(n_end * 9, use_bias=use_bias,
                                   activation=self.params['activation'])(input_)
            encoded = layers.Dense(n_end * 6, use_bias=use_bias,
                                   activation=self.params['activation'])(encoded)
            # encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dense(n_end * 3, use_bias=use_bias, activation=self.params['activation'])(encoded)
            # encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dense(n_end, use_bias=use_bias,
                                   activation=self.params['activation'])(encoded)
        else:
            encoded = layers.Dense(n_end, use_bias=use_bias, activation=self.params['activation'], activity_regularizer=None,
                            kernel_regularizer=None)(input_)
        # Utility Layers

        # Batch Normalization
        encoded = layers.BatchNormalization()(encoded)
        # Soft Thresholding
        encoded = Sparse_ReLU(encoded)
        # Sum To One (ASC)
        encoded = SumToOne(name='abundances')(encoded)

        # Gaussian Dropout
        decoded = layers.GaussianDropout(0.0045)(encoded)

        # Decoder
        decoded = layers.Dense(self.params['n_bands'], activation='linear', name='endmembers',
                               use_bias=False,
                               kernel_constraint=constraints.non_neg())(
            encoded)

        return tf.keras.Model(inputs=input_, outputs=decoded)

    def fit(self, data, plot_every):
        b,h,w,c = data.shape
        data=tf.reshape(data,[-1,c])
        plot_callback = PlotWhileTraining(plot_every, self.params['data'])

        y = self.model.fit(
            x=data,
            y=data,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            callbacks=[plot_callback]
        )
        # Plot loss pictures for training epochs
        loss_values = y.history['loss']
        fig = plt.figure()
        rect = fig.patch
        rect.set_facecolor('white')
        plt.plot(range(1, len(loss_values) + 1), loss_values, 'b-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        save_path = 'DAEUloss_' + self.params["dataset"] + '.png'
        fig.savefig(save_path)
        plt.show()

        return y


    def get_endmembers(self):
        return self.model.layers[len(self.model.layers) - 1].get_weights()[0]

    def get_abundances(self):
        # intermediate_layer_model = tf.keras.Model(
        #     inputs=self.model.input, outputs=self.model.get_layer("abundances").output
        # )
        # abundances = intermediate_layer_model.predict(self.params['data'].array(1))
        # if dataset == 'Samson':
        #     self.params['data'].cols = 96
        #     self.params['data'].rows = 96
        # abundances = np.reshape(abundances,
        #                         [self.params['data'].cols, self.params['data'].rows, self.params['num_endmembers']])
        intermediate_layer_model = tf.keras.Model(
            inputs=self.model.input, outputs=self.model.get_layer("abundances").output
        )
        print(self.params['data'].array(0).shape)
        abundances = intermediate_layer_model.predict(self.params['data'].array(0))
        abundances = np.reshape(abundances,[self.params['data'].cols,self.params['data'].rows,self.params['num_endmembers']])
        return abundances


datasetnames = {
    "sy_data":"sy5",
    "Samson": "Samson",
    "Jasper":"jasper",
    "sy10_data":"sy10",
    "sy20_data":"sy20",
    "Urban":"Urban4"
}
dataset= "sy_data"

hsi = load_HSI(
    "./Datasets/" + datasetnames[dataset] + ".mat"
)


def MSE(y_true,y_pred):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return mse

# Hyperparameters
heads = 5
num_spectra = 1800
batch_size = 6
learning_rate = 0.002
epochs = 20
loss = SAD
plot_every = 0
opt = tf.optimizers.RMSprop(learning_rate=learning_rate)

num_runs =1
results_folder = '\Results'
method_name = 'DAEU_SSABN'

# Hyperparameter dictionary
for ds in [heads]:
    results_folder = './Results'
    params = {
    "num_endmembers": hsi.p,
    "dataset": dataset,
    "batch_size": batch_size,
    "num_spectra": num_spectra,
    "data": hsi,
    "epochs": epochs,
    "n_bands": hsi.bands,
    "GT": hsi.gt,
    "lr": learning_rate,
    "optimizer": opt,
    "loss": loss,
    "activation":layers.LeakyReLU(0.1)
}
    save_folder = results_folder+'/'+method_name+'/'+dataset
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for run in range(1,num_runs+1):
        training_data = list()
        for i in range(0, 20):
            training_data.append(hsi.image[np.newaxis, :])
        training_data = np.concatenate(training_data, axis=0)
        if dataset == 'Samson':
            training_data = tf.pad(training_data, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="SYMMETRIC", name=None,
                                  constant_values=0)
        save_folder = results_folder+'/'+method_name+'/'+dataset
        save_name = dataset+'_run'+str(run)+'.mat'
        save_path = save_folder+'/'+save_name
        start = time.time()
        autoencoder = Autoencoder(params)
        autoencoder.fit(training_data,plot_every)
        endmembers = autoencoder.get_endmembers()
        abundances = autoencoder.get_abundances()
        end = time.time()
        plotEndmembersAndGT(endmembers, hsi.gt)
        SID_plotEndmembersAndGT(endmembers,hsi.gt)
        plotAbundancesSimple(hsi.sgt,'SGT')
        plotAbundancesSimple(abundances,'DAEU_abundance_'+dataset)
        class_rmse, mean_rmse=compute_rmse(hsi.sgt,abundances)
        print(mean_rmse)
        sio.savemat(save_path,{'A':endmembers,'S':abundances})
        cpu_time = end - start
        print('cpu_time:{}'.format(cpu_time))
        del autoencoder