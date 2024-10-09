# 作者:王勇
# 开发时间:2023/9/13 18:35
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
    SID_PlotWhileTraining,
    PlotWhileTraining,
    plotEndmembersAndGT,
    plotAbundancesSimple,
    MSE_PlotWhileTraining,
    MSE_plotEndmembersAndGT,
    load_HSI,
    SID_plotEndmembersAndGT,
    compute_sid,
    compute_rmse,
    compute_sad)
import keras.api.keras.backend as K
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


''''''
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL

# 判断输入数据格式，是channels_first还是channels_last
channel_axis = 1 if K.image_data_format() == "channels_first" else 3

# CAM
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])

# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])

''''''


def SID(y_true, y_pred):
    """计算标准化均方误差"""
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    y_true_mean = tf.reduce_mean(y_true)
    y_true_std = tf.math.reduce_std(y_true)
    sid = mse / (y_true_std**2)
    return sid

def MSE(y_true,y_pred):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return mse

def SAD(y_true, y_pred):
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

class SSAM_layer(tf.keras.Model):
    def __init__(self, filter_sq):
        super().__init__()
        self.filter_sq = filter_sq
        self.avepool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(filter_sq)
        self.relu = LeakyReLU()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):
        b,h,w,c=inputs.shape
        x1=tf.reshape(inputs,(-1,h*w,c))
        dots =tf.einsum('bdi,bdj->bij', x1, x1) * (c**-0.5)
        attn = tf.nn.softmax(dots,axis=-1)
        scale = attn@tf.transpose(x1, perm=[0, 2, 1])  #开始根据attn对x1进行缩放矫正
        scale=tf.transpose(scale, perm=[0, 2, 1])
        scale=tf.reshape(scale,(-1,h,w,c))
        scale = scale + inputs
        return scale

class MSSAM(Layer):
    def __init__(self, num_heads, filter_sq):
        super(MSSAM, self).__init__()
        self.num_heads = num_heads
        self.heads = [SSAM_layer(filter_sq) for _ in range(num_heads)]
    def call(self, inputs):
        head_outputs = [head(inputs) for head in self.heads]
        multihead_output = tf.concat(head_outputs, axis=-1)
        return multihead_output

class Encoder(Model):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.hidden_layer1 = swinT.SwinT(depths=[2], num_heads=[5], window_size=4)
        self.hidden_layer2 = MSSAM(2,16)

        self.asc = SumToOne(params=self.params)

    def call(self, input_patch):
        if params['dataset'] == 'Samson':
            input_patch1 = tf.pad(input_patch, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="SYMMETRIC", name=None,
                                  constant_values=0)
        elif params['dataset'] == "Urban":
            input_patch1 = tf.pad(input_patch, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="SYMMETRIC", name=None,
                                  constant_values=0)
        # if params['dataset'] == "Apex":
        #     input_patch1 = tf.pad(input_patch, [[0, 0], [0, 2], [0, 2], [0, 0]], mode="SYMMETRIC", name=None,
        #                           constant_values=0)
        else:
            input_patch1 = tf.pad(input_patch, [[0, 0], [0, 0], [0, 0], [0, 0]], mode="SYMMETRIC", name=None,
                                  constant_values=0)
        b, h, w, c = input_patch.shape



        code2 = Conv2D(filters=40,
                       kernel_size=1,
                       strides=1,
                       padding='same',
                       use_bias=False)(input_patch1)

        code2 = LeakyReLU()(code2)
        code2 = Dropout(0.2)(code2)

        # code2 = cbam_module(code2)

        code2 = Conv2D(20,1,1,padding='same',use_bias=False)(code2)
        code2 = LeakyReLU()(code2)
        code2 = self.hidden_layer1(code2)

        code11=Conv2D(40,1,1,padding='same',use_bias=False)(input_patch1)
        code11 = BatchNormalization()(code11)
        code11 = LeakyReLU()(code11)
        code11 = Dropout(0.2)(code11)
        code11=self.hidden_layer2(code11)
        code11=Conv2D(16,1,1,padding='same',use_bias=False)(code11)
        code11 = BatchNormalization()(code11)
        code11 = LeakyReLU()(code11)
        code11 = Dropout(0.2)(code11)
        code11=self.hidden_layer2(code11)


        code2 = concatenate([code11, code2], axis=3)
        code2 = Conv2D(self.params['num_endmembers'], 1, 1, use_bias=False)(code2)
        code2 = LeakyReLU()(code2)
        code3 = self.asc(code2)
        code3 = code3[:, 0:h, 0:w, :]

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
        print(B)
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
        # plot_callback = PlotWhileTraining(0, self.params['data'])
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
        save_path = 'loss_' + self.params["dataset"] + '.png'
        fig.savefig(save_path)
        plt.show()

        return y


datasetnames = {
    "Urban":"Urban4",
    "Samson": "Samson",
    "Jasper":"jasper",
    "Sy10_data":"sy10",
    "Sy20_data":"sy20",
    "Sy30_data":"sy30",
    "Sy5_data":"sy5",
    "Apex":"apex"
}
dataset= "Sy5_data"
# Hyperparameters
hsi = load_HSI(
    "./Datasets/" + datasetnames[dataset] + ".mat"
)

# Hyperparameters
batch_size = 2
heads=5
learning_rate = 0.002
epochs=50
scale =1
loss=SAD
opt= tf.optimizers.RMSprop(learning_rate=learning_rate, decay=0)

num_runs =1
results_folder = '\Results'
method_name = '0_SSABN'

for ds in [heads]:
    results_folder = './Results'
    params = {
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
        plotEndmembersAndGT(endmembers, hsi.gt)
        A = hsi.sgt
        A= np.transpose(A, (1, 0, 2))
        abundances = abundances[:,:,(0,2,1,3,4)]
        plotAbundancesSimple(abundances,'abundance_'+dataset)
        plotAbundancesSimple(A,"compute.png")
        a,b=compute_rmse(A,abundances)
        for i in  range(hsi.p):
            print("Class", i + 1, ":", a[i])
        print(b)
        # plotAbundancesSimple(abundances, 'abundance_' + dataset)
        sio.savemat(save_path,{'A':endmembers,'S':abundances})
        cpu_time = end - start
        print('cpu_time:{}'.format(cpu_time))
        del autoencoder
