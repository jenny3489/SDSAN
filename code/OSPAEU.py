import tensorflow as tf
from keras import initializers, constraints, layers, activations, regularizers
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import tensor_shape
from utils import HSI, plotEndmembers,SAD,vca
from utils import plotEndmembersAndGT, plotAbundancesSimple, load_HSI, PlotWhileTraining,SID_plotEndmembersAndGT,MSE_plotEndmembersAndGT
from scipy import io as sio
import os
import numpy as np
from numpy.linalg import inv
import warnings
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def OSP(B,R):
    dots = 0.0
    B = tf.linalg.l2_normalize(B,axis=0)
    for i in range(R):
        for j in range(i+1,R):
            A1 = B[:,i]
            A2 = B[:,j]
            dot = tf.reduce_sum(A1*A2,axis=0)
            dots = dots + dot
    return dots


class SumToOne(layers.Layer):
    def __init__(self, params, **kwargs):
        super(SumToOne, self).__init__(**kwargs)
        self.num_outputs = params['num_endmembers']
        self.params = params

    def l1_regularization(self, x):
        l1 = tf.reduce_sum(tf.pow(tf.abs(x) + 1e-8, 0.7))
        return self.params['l1'] * l1

    def osp_regularization(self, x):
        return self.params['osp'] * OSP(x, self.params['num_endmembers'])

    def call(self, x):
        x = tf.nn.softmax(self.params['scale'] * x)
        self.add_loss(self.l1_regularization(x))
        self.add_loss(self.osp_regularization(x))
        return x

class NonNegLessOne(regularizers.Regularizer):
    def __init__(self, strength):
        super(NonNegLessOne,self).__init__()
        self.strength = strength

    def __call__(self, x):
        neg = tf.cast(x < 0, x.dtype) * x
        greater_one = tf.cast(x>=1.0, x.dtype)*x
        reg = -self.strength * tf.reduce_sum(neg)+self.strength*tf.reduce_sum(greater_one)
        return reg


class HyperLaplacianLoss(object):
    def __init__(self, scale):
        super(HyperLaplacianLoss).__init__()
        self.scale = scale

    def loss(self, X, R):
        fidelity = tf.reduce_mean(tf.pow(tf.abs(X - R) + tf.keras.backend.epsilon(), 0.7), axis=None)
        x = tf.linalg.l2_normalize(X, axis=1)
        r = tf.linalg.l2_normalize(R, axis=1)
        s = X.get_shape().as_list()
        log_cosines = tf.reduce_sum(tf.math.log(tf.reduce_sum(r * x, axis=1) + K.epsilon()))
        return self.scale * fidelity - log_cosines


class Autoencoder(object):
    def __init__(self, params, W=None):
        self.data = params["data"].array(0)
        self.params = params
        self.decoder = layers.Dense(
            units=self.params["n_bands"],
            kernel_regularizer=NonNegLessOne(10),
            activation='linear',
            name="output",
            use_bias=False,
            kernel_constraint=None)
        self.hidden1 = layers.Dense(
            units=self.params["num_endmembers"],
            activation=self.params["activation"],
            name='hidden1',
            use_bias=True
        )
        self.hidden2 = layers.Dense(
            units=self.params["num_endmembers"],
            activation='linear',
            name='hidden2',
            use_bias=True
        )

        self.asc_layer = SumToOne(self.params, name='abundances')
        self.model = self.create_model()
        self.initalize_encoder_and_decoder(W)
        self.model.compile(optimizer=self.params["optimizer"], loss=self.params["loss"])

    def initalize_encoder_and_decoder(self, W):
        if W is None: return
        self.model.get_layer('output').set_weights([W.T])
        self.model.get_layer('hidden1').set_weights([W, np.zeros(self.params["num_endmembers"])])
        W2 = inv(np.matmul(W.T, W))
        self.model.get_layer('hidden2').set_weights([W2, np.zeros(self.params["num_endmembers"])])

    def create_model(self):
        input_features = layers.Input(shape=(self.params["n_bands"],))
        code = self.hidden1(input_features)
        code = self.hidden2(code)
        code = layers.BatchNormalization()(code)
        abunds = self.asc_layer(code)
        output = self.decoder(abunds)

        return tf.keras.Model(inputs=input_features, outputs=output)

    def fix_decoder(self):
        for l in self.model.layers:
            l.trainable = True
        self.model.layers[-1].trainable = False
        self.decoder.trainable = False
        self.model.compile(optimizer=self.params["optimizer"], loss=self.params["loss"])

    def fix_encoder(self):
        for l in self.model.layers:
            l.trainable = True
        self.model.get_layer('hidden1').trainable = False
        self.model.get_layer('hidden2').trainable = False
        self.hidden1.trainable = False
        self.hidden2.trainable = False
        self.model.compile(optimizer=self.params["optimizer"], loss=self.params["loss"])

    def fit(self, data, n):
        plot_callback = PlotWhileTraining(n, self.params['data'])
        return self.model.fit(
            x=data,
            y=data,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            callbacks=[plot_callback]
        )

    def train_alternating(self, data, epochs):
        for epoch in range(epochs):
            self.fix_decoder()
            self.model.fit(x=data, y=data,
                           batch_size=self.params["batch_size"],
                           epochs=2)
            self.fix_encoder()
            self.model.fit(x=data, y=data,
                           batch_size=self.params["batch_size"],
                           epochs=1)

    def get_endmembers(self):
        return self.model.layers[len(self.model.layers) - 1].get_weights()[0]

    def get_abundances(self):
        intermediate_layer_model = tf.keras.Model(
            inputs=self.model.input, outputs=self.model.get_layer("abundances").output
        )
        abundances = intermediate_layer_model.predict(self.data)
        abundances = np.reshape(abundances,
                                [self.params['data'].cols, self.params['data'].rows, self.params['num_endmembers']])

        return abundances


class OutlierDetection(object):
    def __init__(self, image, alpha, threshold):
        self.I = image
        self.alpha = alpha
        self.threshold = threshold

    def get_neighbors(self, row, column):
        n, m, b = self.I.shape
        neighbors_x = np.s_[max(row - 1, 0):min(row + 1, n - 1) + 1]
        neighbors_y = np.s_[max(column - 1, 0):min(column + 1, m - 1) + 1]
        block = np.zeros((3, 3, b))
        block_x = np.s_[max(row - 1, 0) - row + 1:min(row + 1, n - 1) + 1 - row + 1]
        block_y = np.s_[max(column - 1, 0) - column + 1:min(column + 1, m - 1) + 1 - column + 1]
        block[block_x, block_y] = self.I[neighbors_x, neighbors_y, :]
        block = np.reshape(block, (9, -1))
        block = np.delete(block, 5, 0)
        return block

    def d(self, x, y):
        return np.linalg.norm(x - y) ** 2

    def s(self, row, column):
        N = self.get_neighbors(row, column)
        x0 = self.I[row, column, :]
        dists = list(map(lambda x: self.d(x0, x), N))
        return 1 / 8 * sum(list(map(lambda x: np.exp(-x / self.alpha), dists)))

    def create_heatmap(self):
        n, m, b = self.I.shape
        M = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                M[j, i] = self.s(i, j)
        return M

    def get_training_data(self):
        M = self.create_heatmap()
        maxM = np.max(M.flatten())
        indices = np.argwhere(M > self.threshold)
        arr = np.zeros((indices.shape[0], self.I.shape[2]))
        i = 0
        for [r, c] in indices:
            arr[i, :] = self.I[r, c, :]
            i = i + 1
        return [arr, M]

#Dictonary of aliases for datasets. The first string is the key and second is value (name of matfile without .mat suffix)
#Useful when looping over datasets
datasetnames = {
    "Urban": "Urban4",
    "sy10":"sy10",
    "sy20":"sy20",
    "sy30":"sy30"
}
dataset = "sy10"
hsi = load_HSI(
    "./Datasets/" + datasetnames[dataset] + ".mat"
)
data,hmap = OutlierDetection(hsi.image,0.05,0.5).get_training_data()
plt.figure(figsize=(12,12))
plt.imshow(hmap,cmap='gray')
plt.colorbar()

IsOutlierDetection = True

# Hyperparameters
num_endmembers = hsi.p
num_spectra = 2000
batch_size = 15
learning_rate = 0.001
epochs = 13
n_bands = hsi.bands

opt = tf.optimizers.Adam(learning_rate=learning_rate)
activation = 'relu'
l1 = 1.0
osp = 0.5

# hsi.gt=None

if IsOutlierDetection:
    data, hmap = OutlierDetection(hsi.image, 0.05, 0.5).get_training_data()
    num_spectra = data.shape[0]
    batch_size = 256
else:
    data = hsi.array(0)

fid_scale = batch_size
loss = HyperLaplacianLoss(fid_scale).loss

# Hyperparameter dictionary
params = {
    "activation": activation,
    "num_endmembers": num_endmembers,
    "batch_size": batch_size,
    "num_spectra": num_spectra,
    "data": hsi,
    "epochs": epochs,
    "n_bands": n_bands,
    "GT": hsi.gt,
    "lr": learning_rate,
    "optimizer": opt,
    "loss": loss,
    "scale": 1,
    "l1": l1,
    "osp": osp
}

training_data = data[
                np.random.randint(0, data.shape[0], num_spectra), :
                ]
H,W,B = hsi.image.shape
data = hsi.image.reshape(H*W,B)

vca_end = vca(data.T,num_endmembers)[0]
autoencoder = Autoencoder(params,vca_end)
autoencoder.train_alternating(data,epochs)
endmembers = autoencoder.get_endmembers()
abundances = autoencoder.get_abundances()
plotEndmembersAndGT(endmembers, hsi.gt)
SID_plotEndmembersAndGT(endmembers, hsi.gt)
plotAbundancesSimple(abundances,'abund.png')

num_runs = 1
results_folder = './Results'
method_name = 'OSPAEU'

# Dictonary of aliases for datasets. The first string is the key and second is value (name of matfile without .mat suffix)
# Useful when looping over datasets
datasetnames = {
    "Urban": "Urban4",
    "sy10":"sy10",
    "sy20":"sy20",
    "sy30":"sy30"
}

for dataset in [dataset]:
    save_folder = results_folder + '/' + method_name + '/' + dataset
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    dataset_name = dataset

    hsi = load_HSI(
        "./Datasets/" + datasetnames[dataset] + ".mat"
    )
    hsi.image = hsi.image - np.min(hsi.image, axis=2, keepdims=True) + 0.000001  # negative values cause trouble
    data, hmap = OutlierDetection(hsi.image, 0.05, 0.5).get_training_data()

    num_spectra = data.shape[0]
    batch_size = 256
    params['num_spectra'] = num_spectra
    params['data'] = hsi
    params['n_bands'] = hsi.bands

    for run in range(1, num_runs + 1):
        opt = tf.optimizers.Adam(learning_rate=learning_rate)
        params['optimizer'] = opt
        training_data = data[np.random.randint(0, data.shape[0], num_spectra), :]
        save_name = dataset + '_run' + str(run) + '.mat'
        save_path = save_folder + '/' + save_name
        H, W, B = hsi.image.shape
        data = hsi.image.reshape(H * W, B)
        vca_end = vca(data.T, num_endmembers)[0]
        autoencoder = Autoencoder(params, vca_end)
        autoencoder.train_alternating(data, epochs)
        endmembers = autoencoder.get_endmembers()
        abundances = autoencoder.get_abundances()
        plotEndmembersAndGT(endmembers, hsi.gt)
        SID_plotEndmembersAndGT(endmembers,hsi.gt)
        MSE_plotEndmembersAndGT(endmembers,hsi.gt)
        plotAbundancesSimple(abundances, 'abund.png')
        sio.savemat(save_path, {'M': endmembers, 'A': abundances})
