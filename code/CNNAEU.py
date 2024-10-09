import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from numpy.random import seed
seed(2)
import random
random.seed(2)
from keras import Model
from scipy import io as sio
import numpy as np
import os
import swinT
from utils import (
    PlotWhileTraining,
    load_HSI,
    SID_PlotWhileTraining,
    MSE_PlotWhileTraining,
    plotEndmembersAndGT,
    SID_plotEndmembersAndGT,
    MSE_plotEndmembersAndGT,
    plotAbundancesSimple,
    compute_sad,
    compute_rmse)
from keras.layers import (
    Conv2D,
    concatenate,
    LeakyReLU,
    Dense,
    Input,
    Dropout,
    Layer,
    BatchNormalization,
    SpatialDropout2D,
Flatten
)
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def MSE(y_true,y_pred):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return mse

def SAD(y_true, y_pred):
    print(y_true.shape,y_pred.shape)
    A = -tf.keras.losses.cosine_similarity(y_true,y_pred)
    sad = tf.math.acos(A)
    return sad

def array(x):
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        x= sess.run(x)
    return x


class SumToOne(Layer):
    def __init__(self, params, **kwargs):
        super(SumToOne, self).__init__(**kwargs)
        self.params = params

    def call(self, x):
        x = tf.nn.softmax(self.params['scale'] * x)
        return x

class Encoder(Model):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.hidden_layer_one = tf.keras.layers.Conv2D(filters=self.params['e_filters'],
                                                       kernel_size=self.params['e_size'],
                                                       activation=self.params['activation'], strides=1, padding='same',
                                                       kernel_initializer=params['initializer'], use_bias=False)
        self.hidden_layer_two = tf.keras.layers.Conv2D(filters=self.params['num_endmembers'], kernel_size=1,
                                                       activation=self.params['activation'], strides=1, padding='same',
                                                       kernel_initializer=self.params['initializer'], use_bias=False)

        self.asc = SumToOne(params=self.params)

    def call(self, input_patch):
        if params['dataset'] == 'Samson':
            input_patch1 = tf.pad(input_patch, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="SYMMETRIC", name=None,
                                  constant_values=0)
        else:
            input_patch1 = tf.pad(input_patch, [[0, 0], [0, 0], [0, 0], [0, 0]], mode="SYMMETRIC", name=None,
                                  constant_values=0)
        b, h, w, c = input_patch.shape



        code = self.hidden_layer_one(input_patch)            #经历第一个隐藏层+批归一化+dropout层
        code = tf.keras.layers.BatchNormalization()(code)
        code = tf.keras.layers.SpatialDropout2D(0.2)(code)
        code = self.hidden_layer_two(code)                  #经历第二个隐藏层+批归一化+dropout层
        code = tf.keras.layers.BatchNormalization()(code)
        code = tf.keras.layers.SpatialDropout2D(0.2)(code)
        code3 = self.asc(code)

        return code3


class Decoder(Layer):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.output_layer = Dense(units=params["data"].array(0).shape[1],
                                  kernel_regularizer=None,
                                  activation='linear',
                                  name="endmembers",
                                  kernel_constraint=tf.keras.constraints.non_neg(),
                                  use_bias=False)

    def call(self, code):
        B, H, W, C = code.shape
        code = tf.reshape(code, [-1, H * W, C])
        recon = self.output_layer(code)
        recon = tf.reshape(recon, [-1, H, W, self.params["data"].array(0).shape[1]])
        return recon

    def getEndmembers(self):
        w = self.output_layer.get_weights()[0]
        return w


class Autoencoder(object):
    def __init__(self, patches, params):
        self.data = params["data"].image
        self.encoder = Encoder(params)
        self.params = params
        self.decoder = Decoder(params)
        self.H = patches.shape[1]
        self.W = patches.shape[2]
        self.model = self.create_model()
        self.model.compile(optimizer=self.params["optimizer"], loss=self.params["loss"])

    def create_model(self):
        input_a = Input(shape=(self.H, self.W, self.params["data"].array(0).shape[1],))
        abunds = self.encoder(input_a)
        output = self.decoder(abunds)
        return Model(inputs=input_a, outputs=output)

    def get_endmembers(self):
        endmembers = self.decoder.getEndmembers()
        return endmembers

    def get_abundances(self):
        abundances = np.squeeze(self.encoder.predict(np.expand_dims(self.data, 0)))
        return abundances

    def fit(self, patches, n):
        plot_callback = PlotWhileTraining(0, self.params['data'])
        y = self.model.fit(
            x=patches, y=patches,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            callbacks=[plot_callback],
            verbose=1
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
        save_path = 'CNNloss_' + self.params["dataset"] + '.png'
        fig.savefig(save_path)
        plt.show()

        return y


datasetnames = {
    "sy_data":"Sy30",
    "Samson": "Samson",
    "Jasper":"jasper",
    "sy10_data":"sy10",
    "sy5_data":"sy5",
    "Urban":"Urban4"
}
dataset= "sy5_data"
# Hyperparameters
hsi = load_HSI(
    "./Datasets/" + datasetnames[dataset] + ".mat"
)


# Hyperparameters
batch_size = 2
heads=5
learning_rate = 0.002
epochs=25
scale =1
loss=SAD
opt= tf.optimizers.RMSprop(learning_rate=learning_rate, decay=0.0002)

num_runs =1
results_folder = '\Results'
method_name = 'CNN_SSABN'

for ds in [heads]:
    results_folder = './Results'
    params = {
    'e_filters': 48, 'e_size': 3, 'd_filters': 162, 'd_size': 13, 'activation': LeakyReLU(0.02),'initializer': tf.keras.initializers.RandomNormal(0.0, 0.3),
    "dataset":dataset,
    "heads":ds,
    "scale":scale,
    "batch_size": batch_size,
    "data": hsi,
    "epochs": epochs,
    "n_bands": hsi.bands,
    "GT": hsi.gt,
    "lr": learning_rate,
    "optimizer": opt,
    "loss": loss,
    "num_endmembers":hsi.p
    }
    save_folder = results_folder+'/'+method_name+'/'+dataset
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for run in range(1,num_runs+1):
        training_data = list()
        for i in range(0, 20):
            training_data.append(hsi.image[np.newaxis, :])
        training_data = np.concatenate(training_data, axis=0)
        save_folder = results_folder+'/'+method_name+'/'+dataset
        save_name = dataset+'_run'+str(run)+'.mat'
        save_path = save_folder+'/'+save_name
        start = time.time()
        autoencoder = Autoencoder(training_data,params)
        autoencoder.fit(training_data,epochs)
        endmembers = autoencoder.get_endmembers()
        abundances = autoencoder.get_abundances()
        end = time.time()
        SID_plotEndmembersAndGT(endmembers, hsi.gt)
        plotEndmembersAndGT(endmembers,hsi.gt)
        # abundances = abundances[:,:,(1,0,2)]
        A = hsi.sgt
        A= np.transpose(A, (1, 0, 2))
        plotAbundancesSimple(A,"sgt")
        plotAbundancesSimple(abundances,'abundance_'+dataset)
        sio.savemat(save_path,{'A':endmembers,'S':abundances})
        cpu_time = end - start
        print('cpu_time:{}'.format(cpu_time))
        del autoencoder