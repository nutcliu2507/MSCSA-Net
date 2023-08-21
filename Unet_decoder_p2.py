import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation, Add, AveragePooling2D,
                                     BatchNormalization, Conv2D, Conv2DTranspose, Flatten,
                                     Input, MaxPool2D, Reshape, UpSampling2D,
                                     ZeroPadding2D, concatenate, Dropout, SeparableConv2D, DepthwiseConv2D,
                                     GlobalAveragePooling2D, Lambda, Concatenate)
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import tensorflow.keras.backend as backend
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.utils as keras_utils
from Attention import PAM, CAM, local_Se_SA_attention, local_16_Se_SA_attention


def ASPP(tensor):
    dims = K.int_shape(tensor)
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = tensor.shape[channel_axis]
    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]))(tensor)
    y_pool = Conv2D(filters=filters, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = layers.multiply([tensor, y_pool])

    y_1 = Conv2D(filters=filters, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', use_bias=False)(tensor)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = Conv2D(filters=filters, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', use_bias=False)(tensor)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = Conv2D(filters=filters, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', use_bias=False)(tensor)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = Conv2D(filters=filters, kernel_size=3, dilation_rate=18, padding='same',
                  kernel_initializer='he_normal', use_bias=False)(tensor)
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation('relu')(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18])

    y = Conv2D(filters=filters, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y


def Sep_ASPP(tensor):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = tensor.shape[channel_axis]

    y_CA = squeeze_excite_block(tensor)

    y_1 = SeparableConv2D(filters=filters, kernel_size=1, dilation_rate=1, padding='same',
                          kernel_initializer='he_normal', use_bias=False)(tensor)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = SeparableConv2D(filters=filters, kernel_size=3, dilation_rate=6, padding='same',
                          kernel_initializer='he_normal', use_bias=False)(tensor)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = SeparableConv2D(filters=filters, kernel_size=3, dilation_rate=12, padding='same',
                           kernel_initializer='he_normal', use_bias=False)(tensor)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = SeparableConv2D(filters=filters, kernel_size=3, dilation_rate=18, padding='same',
                           kernel_initializer='he_normal', use_bias=False)(tensor)
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation('relu')(y_18)

    y = concatenate([y_CA, y_1, y_6, y_12, y_18])

    y = Conv2D(filters=filters, kernel_size=3, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y


def Sep_3conv(tensor):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = tensor.shape[channel_axis]
    x = Conv2D(ilters=filters, kernel_size=1, padding='same', kernel_initializer='he_normal', use_bias=False)(tensor)
    x_SE = squeeze_excite_block(x)
    x3 = SeparableConv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_regularizer=regularizers.L2(0))(x)
    x5 = SeparableConv2D(filters=filters, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                         kernel_regularizer=regularizers.L2(0))(x)
    x7 = SeparableConv2D(filters=filters, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu',
                         kernel_regularizer=regularizers.L2(0))(x)
    x_c = Concatenate()([x_SE, x3, x5, x7])
    x_c = ConvRelu(filters, 3)(x_c)
    x_c = Add()([tensor, x_c])
    return x_c



def squeeze_excite_block(x, ratio=16):
    init = x
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = layers.Permute((3, 1, 2))(se)

    x = layers.multiply([init, se])
    return x


def S_A_block(x):
    se_output = x
    maxpool_spatial = layers.Lambda(lambda x: backend.max(x, axis=3, keepdims=True))(se_output)
    avgpool_spatial = layers.Lambda(lambda x: backend.mean(x, axis=3, keepdims=True))(se_output)
    max_avg_pool_spatial = layers.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    SA = layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid',
                       kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)
    x = layers.Multiply()([se_output, SA])

    return x


def ConvRelu(filters, kernel_size):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    return layer


def decoder_block(filters, input_tensor):
    x1 = ConvRelu(filters, (1, 1))(input_tensor)
    x1 = carafe(x1, filters, 2, 3, 5)
    x1 = ConvRelu(filters, (1, 1))(x1)
    x = (UpSampling2D((2, 2), data_format='channels_last', interpolation='bilinear'))(input_tensor)
    x = ConvRelu(filters, kernel_size=(3, 3))(x)
    x = Add()([x, x1])

    return x


def carafe(feature_map, cm, upsample_scale, k_encoder, kernel_size):
    """implementation os ICCV 2019 oral presentation CARAFE module"""
    static_shape = feature_map.get_shape().as_list()
    f1 = layers.Conv2D(cm, (1, 1), padding="valid")(feature_map)
    encode_feature = layers.Conv2D(upsample_scale * upsample_scale * kernel_size * kernel_size,
                                   (k_encoder, k_encoder), padding="same")(f1)
    encode_feature = tf.nn.depth_to_space(encode_feature, upsample_scale)
    encode_feature = tf.nn.softmax(encode_feature, axis=-1)

    """encode_feature [B x (h x scale) x (w x scale) x (kernel_size * kernel_size)]"""
    extract_feature = tf.image.extract_patches(feature_map, [1, kernel_size, kernel_size, 1],
                                               strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")

    """extract feature [B x h x w x (channel x kernel_size x kernel_size)]"""
    extract_feature = layers.UpSampling2D((upsample_scale, upsample_scale))(extract_feature)
    extract_feature_shape = tf.shape(extract_feature)
    B = extract_feature_shape[0]
    H = extract_feature_shape[1]
    W = extract_feature_shape[2]
    block_size = kernel_size * kernel_size
    extract_feature = tf.reshape(extract_feature, [B, H, W, block_size, -1])
    extract_feature = tf.transpose(extract_feature, [0, 1, 2, 4, 3])

    """extract feature [B x (h x scale) x (w x scale) x channel x (kernel_size x kernel_size)]"""
    encode_feature = tf.expand_dims(encode_feature, axis=-1)
    upsample_feature = tf.matmul(extract_feature, encode_feature)
    upsample_feature = tf.squeeze(upsample_feature, axis=-1)
    if static_shape[1] is None or static_shape[2] is None:
        upsample_feature.set_shape(static_shape)
    else:
        upsample_feature.set_shape(
            [static_shape[0], static_shape[1] * upsample_scale, static_shape[2] * upsample_scale, static_shape[3]])
    return upsample_feature





# Decoder for UNet is adapted from keras-segmentation
# https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/unet.py
def UNet(f4, f3, f2, f1, output_height, output_width, l1_skip_conn=True, n_classes=6):
    IMAGE_ORDERING = 'channels_last'
    if IMAGE_ORDERING == 'channels_first':
        MERGE_AXIS = 1
    elif IMAGE_ORDERING == 'channels_last':
        MERGE_AXIS = -1
    # ----------------------MARU-net----------------------------------------------------

    x = SeparableConv2D(filters=2048, kernel_size=(7, 7), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.L2(0))(f4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=2048, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.L2(0))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    F5 = x
    F5 = CAM()(F5)
    cam = Conv2D(512, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(F5)
    cam = BatchNormalization(axis=3)(cam)
    F5cam = Activation('relu')(cam)

    # ----------------------FCN--------------------------------------------------
    fc_output = ConvRelu(512, 3)(x)
    o = ConvRelu(512, 3)(f4)
    o = (concatenate([fc_output, o], axis=MERGE_AXIS))
    o = ConvRelu(512, 3)(o)
    o = local_Se_SA_attention(o)
    DF4 = o
    o = Conv2D(512, (3, 3), padding='same', use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING, interpolation='bilinear'))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = ConvRelu(512, 3)(o)
    o = local_Se_SA_attention(o)
    DF3 = o
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING, interpolation='bilinear'))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = ConvRelu(256, 3)(o)
    o = local_Se_SA_attention(o)
    DF2 = o

    # -----------------------------------------------------------------------------------------
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING, interpolation='bilinear'))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))
        o = ConvRelu(128, 3)(o)
        o = local_Se_SA_attention(o)
        DF1 = o
        F4A = Sep_ASPP(DF4)
        DF4 = Add()([DF4, F4A])
        F3A = Sep_ASPP(DF3)
        DF3 = Add()([DF3, F3A])
        F2A = Sep_ASPP(DF2)
        DF2 = Add()([DF2, F2A])
        F1A = Sep_ASPP(DF1)
        DF1 = Add()([DF1, F1A])

        o2 = (concatenate([F5cam, DF4], axis=MERGE_AXIS))
        o2 = ConvRelu(512, 3)(o2)
        o2 = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING, interpolation='bilinear'))(o2)

        o2 = (concatenate([o2, DF3], axis=MERGE_AXIS))
        o2 = ConvRelu(256, 3)(o2)
        o2 = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING, interpolation='bilinear'))(o2)

        o2 = (concatenate([o2, DF2], axis=MERGE_AXIS))
        o2 = ConvRelu(128, 3)(o2)
        o2 = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING, interpolation='bilinear'))(o2)

        o = (concatenate([o2, DF1], axis=MERGE_AXIS))

    # ----------------------output--------------------------------------------------
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = local_16_Se_SA_attention(o)
    o = ConvRelu(64, 3)(o)
    o = (UpSampling2D((4, 4), data_format=IMAGE_ORDERING, interpolation='bilinear'))(o)
    o = Conv2D(n_classes, 1, activation="softmax")(o)


    return o

