import numpy as np
import tensorflow as tf

from keras.layers import (
    Layer,
    Dropout,
    Softmax,
    LayerNormalization,
    Conv2D,
    Activation,
    Dense,
    AvgPool2D,
    Concatenate,
    Lambda,
)
from keras.activations import sigmoid
import collections.abc
from keras import Model, Sequential

def to_2tuple(x):  #其目的是将输入的参数x转换为一个具备两个元素的元组，即长度为2的元组
    if isinstance(x, collections.abc.Iterable):  #检查是否可迭代
        return x
    return (x, x)


class DropPath(Layer): #通过控制drop_path_rate参数来控制是否丢弃网络块的输出。
    def __init__(self, prob):
        super().__init__()
        self.drop_prob = prob #表示丢弃的概率

    def call(self, x, training=None):
        if self.drop_prob == 0. or not training:
            return x  #如果丢弃概率为0或不处于训练，不丢弃任何数据
        keep_prob = 1 - self.drop_prob      #保留概率
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  #获得数据大小
        random_tensor = tf.random.uniform(shape=shape)   #创建一个与输入张量x相同形状的随机张量，
        random_tensor = tf.where(random_tensor < keep_prob, 1, 0)   #随机张量中大于保持概率的设置为0，否则为1
        output = x / keep_prob * random_tensor     #输出 = 输入张量除以保持概率乘以随机张量 （除以保留概率的原因在于：补偿丢失的数据，使得输入与输出的期望一致）
        return output

class TruncatedDense(Dense):   #全连接层，将输入数据与权重矩阵相乘
    def __init__(self, units, use_bias=False):  #units表示输出神经元数量，不适用偏移
        super().__init__(units, use_bias=use_bias)    #构建了一个具有指定输出维度和是否使用偏置项的全连接层

class Mlp(Layer):   #多层感知机，用于引入非线性映射，增加模型的表示能力
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=Activation(tf.nn.gelu), drop=0.): #输入特征维度(神经元数量)，隐藏层神经元数量，输出特征的维度，激活函数的类型(默认为GELU)，不进行dropout操作
        super().__init__()
        out_features = out_features or in_features  #or的作用：没有前者，即是后者
        hidden_features = hidden_features or in_features
        self.fc1 = TruncatedDense(hidden_features)   #TruncatedDense()上述自定义的全连接层
        self.act = act_layer
        self.fc2 = TruncatedDense(out_features)
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)  #将输入张量x通过第一个全连接层fc1进行线性映射（矩阵乘法和偏置项加法）
        x = self.act(x)   #将线性映射的结果通过激活函数act进行非线性变换，通常是GELU函数
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    '''这种结构常用于深度神经网络中，以帮助模型更好地拟合复杂的数据分布。用户可以通过调整in_features、hidden_features、out_features、act_layer和drop等参数来配置Mlp层的结构和行为。'''

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        B：Batch Size
        H：Height
        W：Width
        C：Channel
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    B=-1    # 设置为 -1，这样可以根据输入张量的总元素数量自动计算出批次大小。
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])  #执行分割，H和w分别分割为H(W)/window_size个窗口，分割后的高度、宽度分别为windiow_size。分割后的每个窗口的形状为 (B, window_size, window_size, C)
    # TODO contiguous memory access?
    windows = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), [-1, window_size, window_size, C]) #此处的 -1 指的是在不减少元素总量的情况下，自适应设置维度和大小
    #tf.transpose()函数，对窗口进行重新排序，通过设置 perm 参数，将原来的维度顺序 (B, H // window_size, window_size, W // window_size, window_size, C) 调整为 [B, H // window_size, W // window_size, window_size, window_size, C]，这样可以将窗口之间的像素数据重新排列成连续的形式。
    #最终使用tf.reshape()函数，将重新排列后的张量变换为形状 (num_windows * B, window_size, window_size, C)，其中 num_windows 表示总共划分得到的窗口数量。
    return windows
'''window_partition 函数的主要功能是将输入张量划分为多个窗口，并将这些窗口排列成一个新的张量，以便后续的窗口级别的操作。这种操作在某些视觉任务中非常有用，例如自注意力机制中的窗口注意力操作。'''

@tf.function
def window_reverse(windows, window_size, H, W,C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    #B = int(windows.shape[0] / (H * W / window_size / window_size))
    B=-1
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, C])   #分割时，window最终被分为了(num_windows * B, window_size, window_size, C)，这一步是为了讲扁平化的图像恢复为刚分割的状态
    x = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), [B, H, W, C])
    return x
'''window_reverse函数的功能是将分割后的图像块重新组合为原始图像'''

def SAD(y_true, y_pred):   #计算绝对误差和
    A = -tf.keras.losses.cosine_similarity(y_true,y_pred)
    sad = tf.math.acos(A)
    return sad

def C(x,y):    #调用SAD计算结构相似性指数（SSI）
    val=1.0-SAD(x,y)/np.pi
    return val

class WindowAttention(Layer):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.): #依然不使用bias，qk_scale（query和key的缩放因子）
        super().__init__()
        self.dim = dim      #输入特征通道数
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads    #注意力头的数量
        head_dim = dim // num_heads   #每个注意力头维度
        self.scale = qk_scale or head_dim ** -0.5  #基于头的维度计算缩放因子

        # 定义相对位置偏差参数表
        initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)
        self.relative_position_bias_table = tf.Variable(
            initializer(shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)),  #第一项是垂直方向的相对位置数，第二项为水平
            name="relative_position_bias_table")  # 2*Wh-1 * 2*Ww-1, nH        定义了一个可学习的参数表，用于存储相对位置偏置信息

        # 获取窗口内每个标记的成对相对位置索引
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w))  # 2, Wh, Ww.将垂直和水平坐标堆叠，形成(2,wh,ww)的坐标网格
        coords_flatten = tf.reshape(coords, [2, -1])  # 2, Wh*Ww。压为一维数据
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww，计算相对坐标
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords = relative_coords + [self.window_size[0] - 1, self.window_size[1] - 1]  # shift to start from 0
        relative_coords = relative_coords * [2 * self.window_size[1] - 1, 1]  #进行平移操作，使其可以在0索引处开始
        self.relative_position_index = tf.math.reduce_sum(relative_coords, -1)  # Wh*Ww, Wh*Ww  计算相对位置索引，是相对坐标矩阵在最后一个轴上的求和结果

        self.qkv = TruncatedDense(dim * 3, use_bias=qkv_bias)  #线性变换层，将输入特征映射到查询（query）、键（key）和值（value）空间，因此形状为dim*3
        self.attn_drop = Dropout(attn_drop)
        self.proj = TruncatedDense(dim)    #用于将注意力输出映射回原始特征空间的形状
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(axis=-1)

    def call(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)  切割后的窗口
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        B_=-1
        qkv = tf.transpose(tf.reshape(self.qkv(x), [B_, N, 3, self.num_heads, C // self.num_heads]),  #对输入 x 进行线性变换，得到查询（q）、键（k）和值（v）。(即注意力机制的三个要素)
                           perm=[2, 0, 3, 1, 4])  # [3, B_, num_head, Ww*Wh, C//num_head] 其中 3 表示查询、键和值
        '''tf.unstack()，使得q，k，v可以访问“qkv”张量的各个部件'''
        q, k, v = tf.unstack(qkv)  # make torchscript happy (cannot use tensor as tuple) 这句话的意思是，“让TorchScript满意（不能将张量当作元组使用）”，也即使用torchscript的规则和语法进行编写。

        q = q * self.scale
        attn = tf.einsum('...ij,...kj->...ik', q, k)  #使用爱因斯坦求和符号计算注意力分数。qk交叉关联，形成注意力分数矩阵

        relative_position_bias = tf.reshape(
            tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, [-1])),
            [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
             -1])  # Wh*Ww,Wh*Ww,nH       从预先定义的相对位置偏置表中获取相对位置偏置信息
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])  # nH, Wh*Ww, Wh*Ww 对相对位置信息进行转置，使之匹配注意力矩阵
        attn = attn + relative_position_bias     #注意力分数+bias

        if mask is not None and mask.shape != ():     #检查如果存在掩码，且掩码不为空
            nW = mask.shape[0]  # every window has different mask [nW, N, N]  获得掩码数量
            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N, N]) + mask[:, None, :,
                                                                            :]  # add mask: make each component -inf or just leave it  和掩码相加：将掩码应用于注意力分数，将其中的某些元素设置为 -inf（负无穷）以实现遮挡效果。
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])   #重新调整形状，以匹配softmax计算要求
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)   #不存在掩码，执行softmax，获得标准化的注意力权重

        attn = self.attn_drop(attn)      #dropout，实现稀疏

        x = tf.reshape(tf.transpose(attn @ v, perm=[0, 2, 1, 3]), [B_, N, C])   #计算注意力加权值 attn @ v，然后将结果重新排列为形状 [B_, N, C]，之前计算qk的分数矩阵，现在计算qkv的加权值矩阵
        x = self.proj(x)         #将注意力加权值映射回原始特征空间
        x = self.proj_drop(x)     #dropout，减少过拟合
        return x   #返回值，它包含了通过自注意力层编码的上下文信息

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'   #它返回了层的维度（dim）、窗口大小（window_size）和注意力头的数量（num_heads）。这个字符串通常用于显示模型的摘要信息或调试。

    def flops(self, N):            #这个方法用于估算自注意力层在给定输入序列长度 N 的情况下的浮点操作数（FLOPs，Floating Point Operations）
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim   #计算查询、键和值的线性变换（qkv = self.qkv(x)）所需的FLOPs。
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N  #注意力分数矩阵所需的flops
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)  #注意力加权矩阵所需的flops
        # x = self.proj(x)
        flops += N * self.dim * self.dim          #线性变换所需flops
        return flops



class SwinTransformerBlock(Layer):     #swintransform模块

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=Activation(tf.nn.gelu), norm_layer=LayerNormalization):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  #表示输入图像的分辨率，以元组形式给出（高度，宽度）
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size           #表示窗口的移动大小，用于实现局部注意力，默认为0，表示没有窗口移动。
        self.mlp_ratio = mlp_ratio             #表示MLP（多层感知机）隐藏层的扩展比率，默认为4
        if min(self.input_resolution) <= self.window_size:
            # 如果窗口大小大于输入分辨率，就无法分割窗口，因此需要将shift_size设置为0，并将window_size调整为输入分辨率的最小维度，这是因为窗口大小不能超过输入的大小，否则无法进行分割。
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(epsilon=1e-5)            #标准化层

        # self.ngram_window_partition = NGramWindowPartition(dim,window_size,2,num_heads,shift_size=shift_size)


        self.attn = WindowAttention(                     #窗口注意力层
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.identity   #如果drop>0，则创建dropout层，如果为0，则创建恒等映射函数，此函数相当于不进行操作
        self.norm2 = norm_layer(epsilon=1e-5)  #标准化层
        mlp_hidden_dim = int(dim * mlp_ratio)   #计算隐藏层维度
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) #创建多层感知机层

        if self.shift_size > 0:       #参数大于0，平移
            # 计算 SW-MSA 的注意力掩码
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            '''以上切片分别产生了窗口的起始位置、不进行平移的位置和平移后的位置'''
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt  #每个位置的值，同时cnt也是计数器
                    cnt += 1
            img_mask = tf.constant(img_mask)   #转化为tensorflow张量
            mask_windows = window_partition(img_mask, self.window_size)  # nW(掩码窗口数), window_size, window_size, 1      根据总窗口大小，创建和img_mask相同大小的掩码窗口
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])  #重塑为二维 (nW,window_size*window_size)
            attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]      #根据不同窗口的差异，计算注意力掩码attn_mask，因此attn_mask代表窗口间相对位置
            self.attn_mask = tf.where(attn_mask == 0, -100., 0.)   #tf.where将返回坐标值，此操作将attn_mask中的零值（表示窗口内的相对位置）替换为-100.0，将其他非零值替换为0.0。这样，attn_mask就成为了一个适用于注意力计算的掩码，其中-100.0表示禁止注意力，0.0表示允许注意力。
        else:
            self.attn_mask = None

    def call(self, x):
        '''
            首先，进行输入特征的规范化和形状重塑，以便后续处理。
            然后，根据窗口大小和移动大小对输入特征进行窗口分区，并应用自注意力机制。
            接着，将注意力加权值进行反转和合并，得到输出特征。
            最后，通过MLP和DropPath层来处理输出特征，同时将其与输入特征进行残差连接，得到最终的输出。
            '''
        H, W = self.input_resolution
        B, L, C = x.shape
        B=-1
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, [B, H, W, C])
        '''进行输入特征的规范化和形状重塑，以便后续处理。'''

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=(1, 2))  #x平移，向右下
        else:
            shifted_x = x

        # 窗口分区

        # x_windows = self.ngram_window_partition(x)
        # x_windows = tf.reshape(x_windows,
        #                        [-1, self.window_size * self.window_size, C])  # nW*B, window_size*window_size, C
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C   窗口分割
        x_windows = tf.reshape(x_windows,
                               [-1, self.window_size * self.window_size, C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C     窗口注意力层

        # 合并窗口
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W,C)  # B H' W' C

        # 反循环移动
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=(1, 2))
        else:
            x = shifted_x
        x = tf.reshape(x, [B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)   #shortcut为原始输入x的副本
        '''根据窗口大小和移动大小对输入特征进行窗口分区，并应用自注意力机制。将注意力加权值进行反转和合并，得到输出特征。'''
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:            #此方法用来返回一些类属性，如维度，分辨率等
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):                          #此方法用来计算浮点操作数
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops



class BasicLayer(Layer):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=LayerNormalization):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth        #层的深度

        # build blocks 构建swin模块
        self.blocks = [
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,      #mlp_ratio为感知机比率，决定了隐藏层特征维度(神经元数量)(dim*mlp_ratio)
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)]


    def call(self, x):           #遍历blocks中的所有swin模块，并将输入x传递给每个模块进行处理
        for blk in self.blocks:
            x = blk(x)
        return x

    def extra_repr(self) -> str:          #输出类属性
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):                           #计算全部浮点操作数，即总和每次的浮点数的总和
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class SwinTransformer(Model):

    def __init__(self, out_channels,in_channels, input_resolution,depths=[2], num_heads=[6],
                 window_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=LayerNormalization,
                 **kwargs):
        super().__init__()
        self.dim = in_channels
        self.window_size=window_size
        self.num_layers=len(depths)
        self.input_resolution = tuple([i // self.window_size * self.window_size for i in input_resolution])

        self.mlp_ratio = mlp_ratio
        # stochastic depth
        '''随机深度是一种训练技巧，用于在深层神经网络中随机丢弃一些层，以增强模型的泛化性能。这个概率列表 dpr 用于确定每一层是否要被丢弃，以及丢弃的概率大小。'''
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]  # 随即深度丢弃规则

        # build layers
        self.sequence = Sequential(name="basic_layers_seq")   #创建一个顺序模型
        for i_layer in range(self.num_layers):
            print(self.input_resolution)
            self.sequence.add(BasicLayer(dim=in_channels,
                                         input_resolution=self.input_resolution,
                                         depth=depths[i_layer],
                                         num_heads=num_heads[i_layer],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                         norm_layer=norm_layer))

        # TODO: Check impact of epsilon
        self.norm = norm_layer(epsilon=1e-5)  #epsilon 参数是一个小的常数，用于避免分母为零。
        self.cnn =Conv2D(                     #这个卷积层的作用是进行通道的变换，通常是将模型输出的特征映射到最终的类别数或目标通道数
            filters=out_channels,
            kernel_size=1,
            strides=1, use_bias=False
        )
    def forward_features(self, x):      #向前传播特征
        B, H, W, C = x.shape
        B=-1
        x = tf.reshape(x, [B, H * W,C])
        x = self.sequence(x)
        x = self.norm(x)  # B L C
        x = tf.reshape(x, [B, H, W, C])
        #x = self.cnn(x)
        return x

    def call(self, x):
        x = self.forward_features(x)
        return x

class SwinT(Model):
    def __init__(self,out_channels=1,depths = [2], num_heads = [4],window_size = 4, mlp_ratio = 4., qkv_bias = False, qk_scale = None,
                 drop_rate = 0, attn_drop_rate = 0, drop_path_rate = 0.1,norm_layer = LayerNormalization,):
        super().__init__()
        self.depths = depths
        self.num_heads = num_heads
        self.window_size =window_size
        self.mlp_ratio =mlp_ratio
        self.qkv_bias=qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.out_channels = out_channels

    def forward(self,x):
        # x:[b,c,h,w]
        in_channels=x.shape[3]
        h=x.shape[1]
        w = x.shape[2]
        if self.out_channels == 1:
            self.out_channels = in_channels
        self.SwinT = SwinTransformer(out_channels=self.out_channels,in_channels=in_channels, input_resolution=(h,w),depths=self.depths, num_heads=self.num_heads,
                 window_size=self.window_size, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                 drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop_rate, drop_path_rate=self.drop_path_rate,
                 norm_layer=self.norm_layer)
        return self.SwinT(x)

    def call(self, x):
        x = self.forward(x)
        return x


def window_unpartition(windows, num_windows,ngram):
    """
    Args:
        windows: [B*wh*ww, WH, WW, D]
        num_windows (tuple[int]): The height and width of the window.
    Returns:
        x: [B, ph, pw, D]
    """
    wh,ww = num_windows[0],num_windows[1]
    x = tf.reshape(windows, (-1, wh*ngram, ww*ngram, windows.shape[-1]))
    return x

class NGramContext(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad'):
        super(NGramContext, self).__init__()

        assert padding_mode in ['seq_refl_win_pad', 'zero_pad'], "padding mode should be 'seq_refl_win_pad' or 'zero_pad'!!"

        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.ngram = ngram
        self.padding_mode = padding_mode

        self.unigram_embed = Conv2D(
            filters=dim // 2,
            kernel_size=(self.window_size[0], self.window_size[1]),
            strides=(self.window_size[0], self.window_size[1]),
            padding='valid',
            use_bias=False,
            kernel_initializer='he_normal',
        )
        self.ngram_attn = WindowAttention(dim=dim // 2, num_heads=ngram_num_heads, window_size=(ngram, ngram))
        self.avg_pool = AvgPool2D(pool_size=ngram)
        self.merge = Conv2D(dim, kernel_size=1, strides=1, padding='valid')

    def seq_refl_win_pad(self, x, back=False):
        if self.ngram == 1:
            return x
        x = tf.pad(x, [[0, 0], [0, 0], [self.ngram - 1, 0], [self.ngram - 1, 0]], constant_values=0) if not back else tf.pad(x, [
            [0, 0], [0, 0], [0, self.ngram - 1], [0, self.ngram - 1]], constant_values=0)

        if self.padding_mode == 'zero_pad':
            return x

        if not back:
            start_h, start_w = to_2tuple(-2 * self.ngram + 1)
            end_h, end_w = to_2tuple(-self.ngram)
            input_shape = tf.shape(x)
            pad_height = self.ngram - 1 - (end_h - start_h)
            pad_width = self.ngram - 1 - (end_w - start_w)
            pad_h = tf.zeros([input_shape[0], input_shape[1], pad_height, input_shape[3]])
            pad_w = tf.zeros([input_shape[0], input_shape[1], input_shape[2], pad_width])

            x_padded = tf.concat([x, pad_h], axis=2)
            x_padded = tf.concat([x_padded, pad_w], axis=3)
            x = x_padded
            x = x_padded
        else:
            start_h, start_w = to_2tuple(self.ngram)
            end_h, end_w = to_2tuple(2 * self.ngram - 1)
            input_shape = tf.shape(x)
            pad_height = self.ngram - 1 + (start_h - end_h)
            pad_width = self.ngram - 1 + (start_w - end_w)
            pad_h = tf.zeros([input_shape[0], input_shape[1], pad_height, input_shape[3]])
            pad_w = tf.zeros([input_shape[0], input_shape[1], input_shape[2], pad_width])

            x_padded = tf.concat([x, pad_h], axis=2)
            x_padded = tf.concat([x_padded, pad_w], axis=3)
            x = x_padded
            x = x_padded

        return x

    def sliding_window_attention(self, unigram):
        #unigram    [B,C/2,nw,nh]
        unigram = tf.transpose(unigram, perm=[0, 2, 3, 1])  #[B,nh,nw,c/2 ]


        slide = tf.image.extract_patches(images=unigram,
                                         sizes=[1, self.ngram, self.ngram, 1],
                                         strides=[1, 1, 1, 1],
                                         rates=[1, 1, 1, 1],
                                         padding='VALID')    #[B,24,24,80]

        slide_shape = slide.get_shape().as_list()
        slide = tf.reshape(slide, [-1, 2*(slide_shape[1] - self.ngram + 2), 2*(slide_shape[2] - self.ngram + 2), slide_shape[-1] // 2])  # [B, 2(wh+ngram-2), 2(ww+ngram-2), D/2]
        slide = window_partition(slide, self.ngram)  # [B*wh*ww, ngram, ngram, D/2]  (wh,ww)
        num_windows = (slide_shape[1],slide_shape[2])

        slide = tf.reshape(slide, [-1, self.ngram * self.ngram, self.dim // 2])  # [B*wh*ww, ngram*ngram, D/2]
        # Assuming ngram_attn is your custom attention function
        context = self.ngram_attn(slide)  # [B*wh*ww, ngram, ngram, D/2]

        context = window_unpartition(context, num_windows,self.ngram)  # [B, wh*ngram, ww*ngram, D/2]
        context = self.avg_pool(context)  # [B, D/2, wh, ww]
        return context

    def call(self, x):
        B, ph, pw, D = x.get_shape().as_list()

        unigram = self.unigram_embed(x)  #[B,nW,nH,C/2]

        unigram = tf.transpose(unigram, perm=[0, 3, 1, 2])  # x shape = [B,C/2,nW,nH]

        unigram_forward_pad = self.seq_refl_win_pad(unigram, False)#[B,C/2,nw,nH]
        unigram_backward_pad = self.seq_refl_win_pad(unigram, True)#[B,C/2,nw,nH]


        context_forward = self.sliding_window_attention(unigram_forward_pad)
        context_backward = self.sliding_window_attention(unigram_backward_pad)

        context_bidirect = Concatenate(axis=-1)([context_forward, context_backward])
        context_bidirect = self.merge(context_bidirect)


        return tf.expand_dims(tf.expand_dims(context_bidirect, -2), -2)

    def flops(self, resolutions):
        H, W = resolutions
        wh, ww = H // self.window_size[0], W // self.window_size[1]
        flops = 0
        # unigram embed: conv.weight, conv.bias
        flops += wh * ww * self.window_size[0] * self.window_size[1] * self.dim + wh * ww * self.dim
        # ngram sliding attention (forward & backward)
        flops += 2 * self.ngram_attn.flops(wh * ww)
        # avg pool
        flops += wh * ww * 2 * 2 * self.dim
        # merge concat
        flops += wh * ww * 1 * 1 * self.dim * self.dim
        return flops


class NGramWindowPartition(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, ngram, ngram_num_heads, shift_size=0):
        super(NGramWindowPartition, self).__init__()
        self.window_size = window_size
        self.ngram = ngram
        self.shift_size = shift_size

        self.ngram_context = NGramContext(dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad')

    def call(self, x):
        B, ph, pw, D = x.shape
        wh, ww = ph // self.window_size, pw // self.window_size
        tf.debugging.assert_greater(wh, 0)
        tf.debugging.assert_greater(ww, 0)

        context = self.ngram_context(x)  # [B, wh, D, 1, 1, D]


        windows = tf.reshape(x, (-1, wh, ww, self.window_size, self.window_size, D))
        windows = tf.transpose(windows, perm=(0, 2, 1, 3, 4, 5))  # [B, wh, ww, WH, WW, D]. semi window partitioning

        windows += context  # [B, wh, ww, WH, WW, D]. inject context


        # Cyclic Shift
        if self.shift_size > 0:
            # Re-patchfying
            x = tf.reshape(x,(-1,wh*self.window_size,ww*self.window_size,D))
            # Cyclic shift
            shifted_windows = tf.roll(windows, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])

            # Re-semi window partitioning
            windows = tf.reshape(shifted_windows, [-1, self.window_size, self.window_size, D])

        windows = tf.reshape(windows, [-1,self.window_size,self.window_size, D])

        return windows

    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        flops += self.ngram_context.flops((H, W))
        return flops