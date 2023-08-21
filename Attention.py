import tensorflow.keras.activations
from tensorflow.keras.layers import Activation
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Activation, Add, AveragePooling2D,
                                     BatchNormalization, Conv2D, Conv2DTranspose,
                                     Input, MaxPool2D, Dense, Reshape,
                                     ZeroPadding2D, concatenate, Dropout, SeparableConv2D, DepthwiseConv2D,
                                     GlobalAveragePooling2D, Lambda, Concatenate, Permute, multiply)


class PAM(Layer):
    def __init__(self,
                 beta_initializer=tf.zeros_initializer(),
                 beta_regularizer=None,
                 beta_constraint=None,
                 kernal_initializer='he_normal',
                 kernal_regularizer=None,
                 kernal_constraint=None,
                 **kwargs):
        super(PAM, self).__init__(**kwargs)

        self.beta_initializer = beta_initializer
        self.beta_regularizer = beta_regularizer
        self.beta_constraint = beta_constraint

        self.kernal_initializer = kernal_initializer
        self.kernal_regularizer = kernal_regularizer
        self.kernal_constraint = kernal_constraint

    def build(self, input_shape):
        _, h, w, filters = input_shape

        self.beta = self.add_weight(shape=(1,),
                                    initializer=self.beta_initializer,
                                    name='beta',
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint,
                                    trainable=True)
        # print(self.beta)

        self.kernel_b = self.add_weight(shape=(filters, filters // 8),
                                        initializer=self.kernal_initializer,
                                        name='kernel_b',
                                        regularizer=self.kernal_regularizer,
                                        constraint=self.kernal_constraint,
                                        trainable=True)

        self.kernel_c = self.add_weight(shape=(filters, filters // 8),
                                        initializer=self.kernal_initializer,
                                        name='kernel_c',
                                        regularizer=self.kernal_regularizer,
                                        constraint=self.kernal_constraint,
                                        trainable=True)

        self.kernel_d = self.add_weight(shape=(filters, filters),
                                        initializer=self.kernal_initializer,
                                        name='kernel_d',
                                        regularizer=self.kernal_regularizer,
                                        constraint=self.kernal_constraint,
                                        trainable=True)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()
        _, h, w, filters = input_shape

        b = K.dot(inputs, self.kernel_b)
        c = K.dot(inputs, self.kernel_c)
        d = K.dot(inputs, self.kernel_d)
        vec_b = K.reshape(b, (-1, h * w, filters // 8))
        vec_cT = K.permute_dimensions(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_d = K.reshape(d, (-1, h * w, filters))
        bcTd = K.batch_dot(softmax_bcT, vec_d)
        bcTd = K.reshape(bcTd, (-1, h, w, filters))

        out = self.beta * bcTd + inputs
        # print(self.beta)
        return out

    def get_config(self):
        config = {'beta_initializer': self.beta_initializer,
                  'beta_regularizer': self.beta_regularizer,
                  'beta_constraint': self.beta_constraint,
                  'kernal_initializer': self.kernal_initializer,
                  'kernal_regularizer': self.kernal_regularizer,
                  'kernal_constraint': self.kernal_constraint}
        base_config = super(PAM, self).get_config()
        new_config = list(base_config.items()) + list(config.items())
        return dict(new_config)


class CAM(Layer):
    def __init__(self, **kwargs):
        super(CAM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=tf.zeros_initializer(),
                                     regularizer=None,
                                     constraint=None)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        vec_a = K.reshape(input, (-1, h * w, filters))
        vec_aT = tf.transpose(vec_a, (0, 2, 1))
        aTa = K.batch_dot(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)
        aaTa = K.batch_dot(vec_a, softmax_aTa)
        aaTa = K.reshape(aaTa, (-1, h, w, filters))

        out = self.gamma * aaTa + input
        return out

def LCSA(o):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = o.shape[channel_axis]
    tensor_input = o
    o1, o2 = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(o)
    o11, o12 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(o1)
    o21, o22 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(o2)
    o11se = squeeze_excite_block(o11)
    o11 = Add()([o11, o11se])
    o12se = squeeze_excite_block(o12)
    o12 = Add()([o12, o12se])
    o21se = squeeze_excite_block(o21)
    o21 = Add()([o21, o21se])
    o22se = squeeze_excite_block(o22)
    o22 = Add()([o22, o22se])
    o1 = tf.concat([o11, o12], axis=2)
    # o1 = S_A_block(o1)
    o2 = tf.concat([o21, o22], axis=2)
    # o2 = S_A_block(o2)
    o = tf.concat([o1, o2], axis=1)
    o = Conv2D(filters, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = S_A_block(o)
    o = Add()([o, tensor_input])
    return o

def local_attention(o):
    input_1 = o
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = o.shape[channel_axis]
    o1, o2 = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(o)
    o11, o12 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(o1)
    o21, o22 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(o2)
    o11 = squeeze_excite_block(o11)
    o12 = squeeze_excite_block(o12)
    o21 = squeeze_excite_block(o21)
    o22 = squeeze_excite_block(o22)
    o1 = tf.concat([o11, o12], axis=2)
    o2 = tf.concat([o21, o22], axis=2)
    o = tf.concat([o1, o2], axis=1)
    # o1 = Layer.Concatenate()([o11, o12])
    # o2 = Layer.Concatenate()([o21, o22])
    # o = Layer.Concatenate()([o1, o2])
    # o = Layer.Concatenate()([o, input_1])
    o = Conv2D(filters, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    return o


def squeeze_excite_block(x, ratio=16):
    init = x
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def S_A_block(x):
    se_output = x
    maxpool_spatial = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(se_output)
    avgpool_spatial = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(se_output)
    max_avg_pool_spatial = Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    SA = Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid',
                       kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)
    x = multiply([se_output, SA])

    return x

def local_16_C_S_attention(o):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = o.shape[channel_axis]
    tensor_input = o
    o1, o2, o3, o4 = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 4})(o)
    o11, o12, o13, o14 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 4})(o1)
    o21, o22, o23, o24 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 4})(o2)
    o31, o32, o33, o34 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 4})(o3)
    o41, o42, o43, o44 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 4})(o2)
    # --------o1_SE----------------------------
    o11se = squeeze_excite_block(o11)
    o11 = Add()([o11, o11se])
    o12se = squeeze_excite_block(o12)
    o12 = Add()([o12, o12se])
    o13se = squeeze_excite_block(o13)
    o13 = Add()([o13, o13se])
    o14se = squeeze_excite_block(o14)
    o14 = Add()([o14, o14se])
    # --------o2_SE----------------------------
    o21se = squeeze_excite_block(o21)
    o21 = Add()([o21, o21se])
    o22se = squeeze_excite_block(o22)
    o22 = Add()([o22, o22se])
    o23se = squeeze_excite_block(o23)
    o23 = Add()([o23, o23se])
    o24se = squeeze_excite_block(o24)
    o24 = Add()([o24, o24se])
    # --------o3_SE----------------------------
    o31se = squeeze_excite_block(o31)
    o31 = Add()([o31, o31se])
    o32se = squeeze_excite_block(o32)
    o32 = Add()([o32, o32se])
    o33se = squeeze_excite_block(o33)
    o33 = Add()([o33, o33se])
    o34se = squeeze_excite_block(o34)
    o34 = Add()([o34, o34se])
    # --------o4_SE----------------------------
    o41se = squeeze_excite_block(o41)
    o41 = Add()([o41, o41se])
    o42se = squeeze_excite_block(o42)
    o42 = Add()([o42, o42se])
    o43se = squeeze_excite_block(o43)
    o43 = Add()([o43, o43se])
    o44se = squeeze_excite_block(o44)
    o44 = Add()([o44, o44se])
    # --------左上_dot_cam----------------------------
    olt_1 = tf.concat([o11, o12], axis=2)
    olt_2 = tf.concat([o21, o22], axis=2)
    olt = tf.concat([olt_1, olt_2], axis=1)
    olt_cam = CAM()(olt)
    olt_cam = Conv2D(filters, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(olt_cam)
    olt_cam = BatchNormalization(axis=3)(olt_cam)
    olt_cam = Activation('relu')(olt_cam)
    olt = Add()([olt, olt_cam])
    # --------右上_dot_cam----------------------------
    ort_1 = tf.concat([o13, o14], axis=2)
    ort_2 = tf.concat([o23, o24], axis=2)
    ort = tf.concat([ort_1, ort_2], axis=1)
    ort_cam = CAM()(ort)
    ort_cam = Conv2D(filters, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(ort_cam)
    ort_cam = BatchNormalization(axis=3)(ort_cam)
    ort_cam = Activation('relu')(ort_cam)
    ort = Add()([ort, ort_cam])
    # --------左下_dot_cam----------------------------
    old_1 = tf.concat([o31, o32], axis=2)
    old_2 = tf.concat([o41, o42], axis=2)
    old = tf.concat([old_1, old_2], axis=1)
    old_cam = CAM()(old)
    old_cam = Conv2D(filters, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(old_cam)
    old_cam = BatchNormalization(axis=3)(old_cam)
    old_cam = Activation('relu')(old_cam)
    old = Add()([old, old_cam])
    # --------右下_dot_cam----------------------------
    ord_1 = tf.concat([o33, o34], axis=2)
    ord_2 = tf.concat([o43, o44], axis=2)
    ord = tf.concat([ord_1, ord_2], axis=1)
    ord_cam = CAM()(ord)
    ord_cam = Conv2D(filters, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(ord_cam)
    ord_cam = BatchNormalization(axis=3)(ord_cam)
    ord_cam = Activation('relu')(ord_cam)
    ord = Add()([ord, ord_cam])
    # --------top----------------------------
    ot = tf.concat([olt, ort], axis=2)
    # --------down----------------------------
    od = tf.concat([old, ord], axis=2)
    # --------finish----------------------------
    o = tf.concat([ot, od], axis=1)

    o = Conv2D(filters, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = S_A_block(o)
    o = Add()([o, tensor_input])
    return o

def MSA(tensor, c):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = tensor.shape[channel_axis]
    newtensor = Conv2D(filters=filters/c, kernel_size=1, dilation_rate=1, padding='same',
                          kernel_initializer='he_normal', use_bias=False)(tensor)
    filters2 = newtensor.shape[channel_axis]


    y_CA = squeeze_excite_block(newtensor)

    y_1 = SeparableConv2D(filters=filters2, kernel_size=1, dilation_rate=1, padding='same',
                          kernel_initializer='he_normal', use_bias=False)(newtensor)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = SeparableConv2D(filters=filters2, kernel_size=3, dilation_rate=6, padding='same',
                          kernel_initializer='he_normal', use_bias=False)(newtensor)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = SeparableConv2D(filters=filters2, kernel_size=3, dilation_rate=12, padding='same',
                           kernel_initializer='he_normal', use_bias=False)(newtensor)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = SeparableConv2D(filters=filters2, kernel_size=3, dilation_rate=18, padding='same',
                           kernel_initializer='he_normal', use_bias=False)(newtensor)
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation('relu')(y_18)

    y = concatenate([y_CA, y_1, y_6, y_12, y_18])

    y = Conv2D(filters=filters, kernel_size=3, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y