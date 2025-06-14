####### Importing Libraries
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# plt.switch_backend('agg')
import tensorflow as tf
import os
import gc
import math
# import pydot
# from sklearn.utils import shuffle

####### Loading Dataset

####### Loading Dataset

####### Loading Dataset

###### RDI Sequence
X_train_rdi = np.load('Dataset/X_train_rdi_soli.npz', allow_pickle=True)['arr_0']
X_dev_rdi = np.load('Dataset/X_dev_rdi_soli.npz', allow_pickle=True)['arr_0']

###### RAI Sequence
X_train_rai = np.load('Dataset/X_train_rai_soli.npz', allow_pickle=True)['arr_0']
X_dev_rai = np.load('Dataset/X_dev_rai_soli.npz', allow_pickle=True)['arr_0']
y_train = np.load('Dataset/y_train_soli.npz', allow_pickle=True)['arr_0']
y_dev = np.load('Dataset/y_dev_soli.npz', allow_pickle=True)['arr_0']
###### Converting Labels to Categorical Format
y_train_onehot = tf.keras.utils.to_categorical(y_train)
y_dev_onehot = tf.keras.utils.to_categorical(y_dev)


####### Model Makingclenv

####### TEA Module
####### TEA - Temporal Excitation and Aggregation Network

###### Motion Excitation (ME) Module

class TEA_ME(tf.keras.layers.Layer):
    """ TEA Module's Motion Excitation Block for Motion Modelling """

    def __init__(self, reduction_factor, num_channels):
        #### Defining Essentials
        super().__init__()
        self.reduction_factor = reduction_factor  # Reduction Factor for Reducing Conv
        self.num_channels = num_channels  # Number of Channels in the Input

        #### Defining Layers
        red_val = int(self.num_channels // self.reduction_factor)
        self.conv_red = tf.keras.layers.Conv2D(filters=red_val, kernel_size=(1, 1), padding='same',
                                               activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

        self.conv_transform = tf.keras.layers.Conv2D(filters=red_val, kernel_size=(3, 3), padding='same',
                                                     groups=red_val, activation='relu',
                                                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))

        self.conv_exp = tf.keras.layers.Conv2D(filters=self.num_channels, kernel_size=(1, 1), padding='same',
                                               activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'reduction_factor': self.reduction_factor,
            'num_channels': self.num_channels
        })
        return config

    def call(self, X):
        """
        Implementation of ME Module

        INPUTS:-
        1) X : Input Tensor of Shape [N,T,H,W,C] (Implementation involves 'Channel Last' Strategy)

        OUTPUTS:-
        1) X_o : Tensor of shape [N,T,H,W,C]

        """

        #### Extracting Input Dimensions
        N = (X.shape[0])  # Batch Size
        T = (X.shape[1])  # Total Frames in the Signal
        H = (X.shape[2])  # Height of the Frame
        W = (X.shape[3])  # Width of the Frame
        C = self.num_channels  # Number of Channels in the Input

        #### Reduction of Channel Dimensions
        X_red = self.conv_red(X)

        #### Motion Modelling
        X_red_M1 = X_red[:, :-1, :, :, :]  # Taking the X_red till the penultimate frame
        X_red_M2 = X_red[:, 1:, :, :, :]  # Taking the X_red from the second frame till the end

        X_transform = self.conv_transform(X_red_M2)  # Channel-Wise Convolution

        M = tf.keras.layers.Add()([X_transform, -X_red_M1])  # Action Modelling
        M = tf.keras.layers.ZeroPadding3D(((1, 0), (0, 0), (0, 0)))(M)  # Adding M(T) = 0 Frame

        #### Global Average Pooling
        Ms = tf.keras.layers.AveragePooling3D(pool_size=(1, H, W))(M)

        #### Convolution for Channel Expansion
        Ms_expanded = self.conv_exp(Ms)

        #### Motion Attentive Weights Computation
        A = 2 * (tf.keras.activations.sigmoid(Ms_expanded)) - 1

        #### Output Creation
        X_bar = tf.math.multiply(X, A)  # Channel-wise multiplication of attentive weights
        X_o = tf.keras.layers.Add()([X, X_bar])  # Residual Connection

        return X_o


###### Multiple Temporal Aggregation (MTA) Module

class TEA_MTA(tf.keras.layers.Layer):
    def __init__(self, N, T, H, W, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.N = N
        self.T = T
        self.H = H
        self.W = W
        self.split_factor = self.num_channels // 4

        # Replacing grouped Conv1D with standard Conv1D layers
        self.temp_conv1_layers = [tf.keras.layers.Conv1D(
            filters=1, kernel_size=3, padding='same', activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)) for _ in range(self.split_factor)]

        self.temp_conv2_layers = [tf.keras.layers.Conv1D(
            filters=1, kernel_size=3, padding='same', activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)) for _ in range(self.split_factor)]

        self.temp_conv3_layers = [tf.keras.layers.Conv1D(
            filters=1, kernel_size=3, padding='same', activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)) for _ in range(self.split_factor)]

        self.conv_spa_1 = tf.keras.layers.Conv2D(filters=self.split_factor, kernel_size=(3, 3), padding='same',
                                                 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv_spa_2 = tf.keras.layers.Conv2D(filters=self.split_factor, kernel_size=(3, 3), padding='same',
                                                 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv_spa_3 = tf.keras.layers.Conv2D(filters=self.split_factor, kernel_size=(3, 3), padding='same',
                                                 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_channels': self.num_channels,
            'N': self.N,
            'T': self.T,
            'H': self.H,
            'W': self.W
        })
        return config

    def grouped_conv1d(self, x, conv_layers):
        # x: [B, L, C], conv_layers: list of Conv1D layers for each channel
        x_splits = tf.split(x, num_or_size_splits=self.split_factor, axis=-1)
        out_splits = [conv_layer(split) for conv_layer, split in zip(conv_layers, x_splits)]
        return tf.concat(out_splits, axis=-1)

    def call(self, X):
        batch_size = tf.shape(X)[0]
        T = self.T
        H = self.H
        W = self.W
        C = self.num_channels
        split_factor = self.split_factor

        # Split input channels into 4 parts
        Xi_0, Xi_1, Xi_2, Xi_3 = tf.split(X, num_or_size_splits=4, axis=-1)

        # First branch: pass through
        Xo_0 = Xi_0

        # Second branch: temporal + spatial attention
        Xi_1 = tf.keras.layers.Add()([Xo_0, Xi_1])
        Xi_1_reshaped_temp = tf.reshape(Xi_1, [batch_size * T, H * W, split_factor])
        Xi_1_temp = self.grouped_conv1d(Xi_1_reshaped_temp, self.temp_conv1_layers)
        Xi_1_reshaped_spa = tf.reshape(Xi_1_temp, [batch_size * T, H, W, split_factor])
        Xo_1 = self.conv_spa_1(Xi_1_reshaped_spa)
        Xo_1 = tf.reshape(Xo_1, [batch_size, T, H, W, split_factor])

        # Third branch
        Xi_2 = tf.keras.layers.Add()([Xo_1, Xi_2])
        Xi_2_reshaped_temp = tf.reshape(Xi_2, [batch_size * T, H * W, split_factor])
        Xi_2_temp = self.grouped_conv1d(Xi_2_reshaped_temp, self.temp_conv2_layers)
        Xi_2_reshaped_spa = tf.reshape(Xi_2_temp, [batch_size * T, H, W, split_factor])
        Xo_2 = self.conv_spa_2(Xi_2_reshaped_spa)
        Xo_2 = tf.reshape(Xo_2, [batch_size, T, H, W, split_factor])

        # Fourth branch
        Xi_3 = tf.keras.layers.Add()([Xo_2, Xi_3])
        Xi_3_reshaped_temp = tf.reshape(Xi_3, [batch_size * T, H * W, split_factor])
        Xi_3_temp = self.grouped_conv1d(Xi_3_reshaped_temp, self.temp_conv3_layers)
        Xi_3_reshaped_spa = tf.reshape(Xi_3_temp, [batch_size * T, H, W, split_factor])
        Xo_3 = self.conv_spa_3(Xi_3_reshaped_spa)
        Xo_3 = tf.reshape(Xo_3, [batch_size, T, H, W, split_factor])

        # Concatenate all outputs
        Xo = tf.keras.layers.Concatenate(axis=-1)([Xo_0, Xo_1, Xo_2, Xo_3])

        return Xo


####### CT-Module

####### Channel Tensorization (CT-Module)

class CT_Module(tf.keras.layers.Layer):
    """ 3D Tensor Separable Convolution """

    def __init__(self, T, H, W, C):
        ##### Defining Instatiations
        super().__init__()
        self.T = T  # Total number of Frames
        self.H = H  # Height of the Input
        self.W = W  # Width of the Input
        self.C = C  # Channels in the Input

        K = int(math.log2(C))
        k1_dim = int(K / 2)
        self.k1 = int(2 ** (k1_dim))  # Sub Dimension 1
        self.k2 = int(2 ** (K - k1_dim))  # Sub Dimension 2

        ##### Defining Layers
        self.conv_k1 = tf.keras.layers.Conv3D(filters=self.C, kernel_size=(3, 3, 3), padding='same',
                                              activation='linear', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv_k2 = tf.keras.layers.Conv3D(filters=self.C, kernel_size=(3, 3, 3), padding='same',
                                              activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'T': self.T,
            'H': self.H,
            'W': self.W,
            'C': self.C,
            'k1': self.k1,
            'k2': self.k2
        })
        return config

    def call(self, X0):
        """
        Implementation of Tensorization Module

        INPUTS:
        1) X0 : Input Tensor of shape (T, H, W, C)

        OUTPUTS:
        1) X2 : Output Tensor of shape (T, H, W, C)
        """
        # Validate input shape
        if X0.shape[-1] != self.C:
            raise ValueError(f"Input channels ({X0.shape[-1]}) do not match expected channels ({self.C}).")

        # Reshape input to split channels
        X0 = tf.keras.layers.Reshape((self.T, self.H, self.W, self.k1 * self.k2))(X0)

        # First Sub-Convolution
        X1 = self.conv_k1(X0)  # Conv3D expects 5D tensor

        # Second Sub-Convolution
        X2 = self.conv_k2(X1)

        # Reshape back to original input dimensions
        X2 = tf.keras.layers.Reshape((self.T, self.H, self.W, self.C))(X2)

        return X2


####### (2+1)D Convolution

####### (2+1)D Convolutional Layer

class two_plus_oneDConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_dims, H, W, C, T):
        super().__init__()
        self.filters = filters
        self.kernel_dims = kernel_dims
        self.H = H
        self.W = W
        self.C = C
        self.T = T

        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(self.kernel_dims, self.kernel_dims),
            padding='same',
            activation='linear',
            depth_multiplier=1
        )

        self.pointwise_conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )

        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_dims,
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )

    def build(self, input_shape):
        # Apply the regularizer after the layer is built
        if hasattr(self.depthwise_conv, 'depthwise_kernel_regularizer'):
            self.depthwise_conv.depthwise_kernel_regularizer = tf.keras.regularizers.l2(1e-5)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_dims': self.kernel_dims,
            'H': self.H,
            'W': self.W,
            'C': self.C,
            'T': self.T
        })
        return config

    def call(self, X):
        X_reshaped = tf.reshape(X, [-1, self.H, self.W, self.C])
        X_conv2d = self.depthwise_conv(X_reshaped)
        X_conv2d = self.pointwise_conv(X_conv2d)

        X_conv2d = tf.reshape(X_conv2d, [-1, self.T, self.H * self.W, self.filters])
        X_conv2d = tf.transpose(X_conv2d, perm=[0, 2, 1, 3])
        X_flat = tf.reshape(X_conv2d, [-1, self.T, self.filters])
        X_conv1d = self.conv1d(X_flat)

        X_conv1d = tf.reshape(X_conv1d, [-1, self.H * self.W, self.T, self.filters])
        X_conv1d = tf.transpose(X_conv1d, perm=[0, 2, 1, 3])
        X_o = tf.reshape(X_conv1d, [-1, self.T, self.H, self.W, self.filters])

        return X_o




class Cross_MSECA_Module(tf.keras.layers.Layer):
    """ Implementation of 3D MSECA(Multi-Scale Efficient Channel Attention) Module """

    def __init__(self, T, H, W, C, k):
        ##### Defining Essentials

        super().__init__()
        self.T = T
        self.H = H
        self.W = W
        self.C = C
        self.k = k  # Kernel dims

        ##### Defining Layers

        #### Adaptive Kernel Size Selection
        # t = int(abs((math.log2(self.C)+self.b)/self.gamma))
        # k = t if t%2 else t+1

        #### Convolution Layers
        self.conv_k1 = tf.keras.layers.Conv1D(filters=1, kernel_size=self.k,
                                              padding='same', activation='linear',
                                              use_bias=False)
        self.conv_k2 = tf.keras.layers.Conv1D(filters=1, kernel_size=(self.k) ** 2,
                                              padding='same', activation='linear',
                                              use_bias=False)
        self.conv_k3 = tf.keras.layers.Conv1D(filters=1, kernel_size=(self.k) ** 3,
                                              padding='same', activation='linear',
                                              use_bias=False)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'T': self.T,
            'H': self.H,
            'W': self.W,
            'C': self.C,
            'k': self.k
        })
        return config

    def call(self, X_in):
        """
        Implemetation of MSECA Module

        INPUTS
        1)X_in : Input of Shape - (,T,H,W,C)

        OUTPUTS
        1)X_mseca : Attentioned Output

        """
        X_in_reshaped = tf.keras.layers.Reshape((-1, self.C))(X_in) # Reshaping the Input to 2D
        X = tf.keras.layers.GlobalAveragePooling3D()(X_in)  # Global Average Pooling
        X = tf.keras.layers.Reshape((self.C, 1))(X)  # Resizing Dimensions for 1D Convolution

        X_k1 = self.conv_k1(X)  # Attention Weight Computation with kernel k1
        X_k2 = self.conv_k2(X)  # Attention Weight Computation with kernel k2
        X_k3 = self.conv_k3(X)  # Attention Weight Computation with kernel k3

        X = tf.keras.layers.Add()([X_k1, X_k2, X_k3])  # Adding Multiscale Information
        X_reshaped = tf.keras.layers.Reshape((1, self.C))(X)  # Reshaping the Multiscale Information

        X_map = tf.linalg.matmul(X, X_reshaped)  # Channel map with one-to-one correspondence
        X_attn_map = tf.keras.layers.Softmax(axis=2)(X_map)  # Softmax activation

        X_mseca = tf.linalg.matmul(X_in_reshaped, X_attn_map)  # Activating input by corresponding attention
        X_mseca = tf.keras.layers.Reshape((self.T, self.H, self.W, self.C))(X_mseca)  # Reshaping final output

        return X_mseca

    ###### Arc Loss


class ArcFace(tf.keras.layers.Layer):

    def __init__(self, n_classes, s, m, regularizer):
        super().__init__()
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer
        })
        return config

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True
                                 )

    def call(self, inputs):
        x, y = inputs
        c = tf.keras.backend.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(
            tf.keras.backend.clip(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)
#### Mamba defination
class MambaBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = None
        self.linear_B = None
        self.linear_C = None
        self.out_proj = None
        self.A = None

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError("Channel dimension must be defined for MambaBlock.")
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.linear_B = tf.keras.layers.Dense(self.hidden_dim)
        self.linear_C = tf.keras.layers.Dense(self.hidden_dim)
        self.out_proj = tf.keras.layers.Dense(self.hidden_dim)

        self.A = self.add_weight(
            name="A",
            shape=(self.hidden_dim,),
            initializer="zeros",
            trainable=True
        )




##### Gmamba Defination
class GMamba(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.mamba = MambaBlock(hidden_dim)

    def call(self, x):
        d_seq = x[:, 1:, :, :, :] - x[:, :-1, :, :, :]
        zero_pad = tf.zeros_like(x[:, :1, :, :, :])
        d_seq = tf.concat([zero_pad, d_seq], axis=1)

        d_seq_norm = self.norm(d_seq)

        # Ensure build is called manually (required in Functional API if shape is dynamic)
        if not self.mamba.built:
            self.mamba.build(d_seq_norm.shape)

        out = self.mamba(d_seq_norm)
        return d_seq + out



### Lmamba defination
class LMamba(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mamba = MambaBlock(hidden_dim)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)

    def call(self, x):
        # Ensure mamba is built
        if not self.mamba.built:
            self.mamba.build(x.shape)

        s = self.mamba(x)
        s_norm = self.norm(s)
        return x + s_norm



####### Model Training

####### Defining Layers and Model

###### Defining Layers

##### Input Shapes
T = 40
H = 32
W = 32
C_rdi = 4
C_rai = 1

##### Convolutional Layers

#### RDI
conv_up1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), padding='same',
                                  activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
conv11_rdi = CT_Module(40, 32, 32, 32)
conv12_rdi = CT_Module(40, 32, 32, 32)
conv13_rdi = CT_Module(40, 32, 32, 32)

conv_up2 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 1), padding='same',
                                  activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
conv21_rdi = CT_Module(40, 32, 32, 64)
conv22_rdi = CT_Module(40, 32, 32, 64)
conv23_rdi = CT_Module(40, 32, 32, 64)

#### RAI
conv11_rai = two_plus_oneDConv(32, 3, 32, 32, 1, 40)
conv12_rai = two_plus_oneDConv(32, 3, 32, 32, 32 + 1, 40)
conv13_rai = two_plus_oneDConv(32, 3, 32, 32, 32 + 32 + 1, 40)

conv21_rai = two_plus_oneDConv(64, 3, 32, 32, 32, 40)
conv22_rai = two_plus_oneDConv(64, 3, 32, 32, 64 + 32, 40)
conv23_rai = two_plus_oneDConv(64, 3, 32, 32, 160, 40)

##### Initialize GMamba and LMamba
gmamba_layer = GMamba()
lmamba_layer = LMamba(hidden_dim=128)


##### ArcFace Loss
arc_logit_layer = ArcFace(11, 30.0, 0.3, tf.keras.regularizers.l2(1e-4))

###### Defining Model

##### Input Layer
Input_Layer_rdi = tf.keras.layers.Input(shape=(T, H, W, C_rdi))
Input_Layer_rai = tf.keras.layers.Input(shape=(T,H,W,C_rai))
Input_Labels = tf.keras.layers.Input(shape=(11,))
# Input_Layer_rdi = tf.keras.layers.Input(shape=(None, H, W, C_rdi))
# Input_Layer_rdi = tf.keras.layers.Input(shape=(40, 32, 32, 4))  # Fix temporal dimension
# Input_Labels = tf.keras.layers.Input(shape=(11,))


##### Conv Layers

#### RDI
### Tensorized Residual Block - 1
print("input_layer_rdi",Input_Layer_rdi.shape)
conv_up1 = conv_up1(Input_Layer_rdi)
print("conv_up1 size",conv_up1.shape)
conv11_rdi = conv11_rdi(conv_up1)
conv12_rdi = conv12_rdi(conv11_rdi)
print("Before Add: conv12_rdi", conv12_rdi.shape, "conv_up1", conv_up1.shape)
conv12_rdi = tf.keras.layers.Add()([conv12_rdi, conv_up1])
print("After Add: conv12_rdi", conv12_rdi.shape, "conv_up1", conv_up1.shape)
conv12_rdi = tf.keras.layers.Add()([conv12_rdi, conv_up1])

conv12_rdi = tf.keras.layers.Add()([conv12_rdi, conv_up1])
conv13_rdi = conv13_rdi(conv12_rdi)
conv13_rdi = tf.keras.layers.Add()([conv13_rdi, conv11_rdi])

### Tensorized Residual Block - 2
conv_up2 = conv_up2(conv13_rdi)
conv21_rdi = conv21_rdi(conv_up2)
conv22_rdi = conv22_rdi(conv21_rdi)
conv22_rdi = tf.keras.layers.Add()([conv22_rdi, conv_up2])
conv23_rdi = conv23_rdi(conv22_rdi)
conv23_rdi = tf.keras.layers.Add()([conv23_rdi, conv21_rdi])

#### RAI
### Dense Block - 1
conv11_rai = conv11_rai(Input_Layer_rai)
conv11_rai = tf.keras.layers.Concatenate(axis=-1)([conv11_rai,Input_Layer_rai])
conv12_rai = conv12_rai(conv11_rai)
conv12_rai = tf.keras.layers.Concatenate(axis=-1)([conv12_rai,conv11_rai])
conv13_rai = conv13_rai(conv12_rai)

### Dense Block - 2
conv21_rai = conv21_rai(conv13_rai)
conv21_rai = tf.keras.layers.Concatenate(axis=-1)([conv21_rai,conv13_rai])
conv22_rai = conv22_rai(conv21_rai)
conv22_rai = tf.keras.layers.Concatenate(axis=-1)([conv22_rai,conv21_rai])
conv23_rai = conv23_rai(conv22_rai)

#### Concatenation Operation
X_concat= tf.keras.layers.Concatenate(axis=-1)([conv23_rdi,conv23_rai])

# Pass through variation-aware Mamba modules
X_global = gmamba_layer(X_concat)   # GMamba (variation-based)
X_local  = lmamba_layer(X_concat)   # LMamba (local residual refinement)

X_gmamba_lmamba_add = tf.keras.layers.Add()([X_global, X_local])
print(X_gmamba_lmamba_add.shape)

gap_op = tf.keras.layers.GlobalAveragePooling3D()(X_gmamba_lmamba_add)
dense1 = tf.keras.layers.Dense(256, activation='relu')(gap_op)
dropout1 = tf.keras.layers.Dropout(rate=0.2)(dense1)

### ArcFace Output Layer
dense2 = tf.keras.layers.Dense(256, kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout1)
##dense2 = tf.keras.layers.BatchNormalization()(dense2)
dense3 = arc_logit_layer(([dense2, Input_Labels]))

###### Compiling Model
model = tf.keras.models.Model(inputs=[Input_Layer_rdi, Input_Layer_rai,Input_Labels], outputs=dense3)
model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

###### Training the Model
history = model.fit(
    [X_train_rdi, X_train_rai,y_train_onehot], y_train_onehot,
    epochs=30,
    batch_size=2,
    validation_data=([X_dev_rdi, X_dev_rai,y_dev_onehot], y_dev_onehot),
    validation_batch_size=2)

##### Saving Training Metrics
np.save('exp_1_gamba_lmamba_history.npy', history.history)

# Save only the architecture
model_json = model.to_json()
with open("exp_1_gamba_lmamba_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Save only the weights
model.save_weights("exp_1_gamba_lmamba.weights.h5")