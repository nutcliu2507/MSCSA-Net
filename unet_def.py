import tensorflow as tf
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv2D,
                                     UpSampling2D, concatenate, SeparableConv2D)
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
from Attention import (squeeze_excite_block, LCSA, CAM, local_16_C_S_attention, MSA)



def ConvRelu(filters, kernel_size):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    return layer

def carafe(feature_map, cm, upsample_scale, k_encoder, kernel_size):
    static_shape = feature_map.get_shape().as_list()
    f1 = layers.Conv2D(cm, (1, 1), padding="valid")(feature_map)
    encode_feature = layers.Conv2D(upsample_scale * upsample_scale * kernel_size * kernel_size,
                                   (k_encoder, k_encoder), padding="same")(f1)
    encode_feature = tf.nn.depth_to_space(encode_feature, upsample_scale)
    encode_feature = tf.nn.softmax(encode_feature, axis=-1)

    extract_feature = tf.image.extract_patches(feature_map, [1, kernel_size, kernel_size, 1],
                                               strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")

    extract_feature = layers.UpSampling2D((upsample_scale, upsample_scale))(extract_feature)
    extract_feature_shape = tf.shape(extract_feature)
    B = extract_feature_shape[0]
    H = extract_feature_shape[1]
    W = extract_feature_shape[2]
    block_size = kernel_size * kernel_size
    extract_feature = tf.reshape(extract_feature, [B, H, W, block_size, -1])
    extract_feature = tf.transpose(extract_feature, [0, 1, 2, 4, 3])

    encode_feature = tf.expand_dims(encode_feature, axis=-1)
    upsample_feature = tf.matmul(extract_feature, encode_feature)
    upsample_feature = tf.squeeze(upsample_feature, axis=-1)
    if static_shape[1] is None or static_shape[2] is None:
        upsample_feature.set_shape(static_shape)
    else:
        upsample_feature.set_shape(
            [static_shape[0], static_shape[1] * upsample_scale, static_shape[2] * upsample_scale, static_shape[3]])
    return upsample_feature

def Upsample_twice(x, up_size):
    init = x
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    u1 = (UpSampling2D((up_size, up_size), data_format='channels_last', interpolation='bilinear'))(x)
    u1 = squeeze_excite_block(u1)
    u2 = carafe(x, filters, up_size, 3, 5)
    u2 = squeeze_excite_block(u2)
    x = Add()([u1, u2])
    return x

def UNet(f4, f3, f2, f1, output_height, output_width, l1_skip_conn=True, n_classes=6):
    IMAGE_ORDERING = 'channels_last'
    if IMAGE_ORDERING == 'channels_first':
        MERGE_AXIS = 1
    elif IMAGE_ORDERING == 'channels_last':
        MERGE_AXIS = -1
    # --------------------------------------------------------------------------

    x = SeparableConv2D(filters=512, kernel_size=(7, 7), strides=(1, 1), padding='same')(f4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    F5 = x
    F5 = CAM()(F5)
    F5cam = ConvRelu(512, 3)(F5)

    o = ConvRelu(512, 3)(f4)
    o = LCSA(o)
    o = (concatenate([o, F5], axis=MERGE_AXIS))
    o = ConvRelu(512, 3)(o)
    DF4 = o
    o = Upsample_twice(o, 2)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = ConvRelu(256, 3)(o)
    o = LCSA(o)
    DF3 = o
    o = Upsample_twice(o, 2)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = ConvRelu(128, 3)(o)
    o = LCSA(o)
    DF2 = o
    o = Upsample_twice(o, 2)
    o = (concatenate([o, f1], axis=MERGE_AXIS))
    o = ConvRelu(64, 3)(o)
    o = LCSA(o)
    DF1 = o
    # ---------------------------------------------------------------------------
    F4A = MSA(DF4, 8)
    o = (concatenate([F4A, F5cam], axis=MERGE_AXIS))
    tF4 = ConvRelu(256, 1)(o)
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING, interpolation='bilinear'))(tF4)
    o = (concatenate([o, DF3], axis=MERGE_AXIS))
    DF3 = ConvRelu(256, 3)(o)
    #------------------------------------------------
    F3A = MSA(DF3, 8)
    tF3 = ConvRelu(128, 1)(F3A)
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING, interpolation='bilinear'))(tF3)
    o = (concatenate([o, DF2], axis=MERGE_AXIS))
    DF2 = ConvRelu(128, 1)(o)
    #------------------------------------------------
    F2A = MSA(DF2, 8)
    tF2 = ConvRelu(64, 1)(F2A)
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING, interpolation='bilinear'))(tF2)
    o = (concatenate([o, DF1], axis=MERGE_AXIS))
    DF1 = ConvRelu(64, 1)(o)
    # -----------------------------------------------------------------------------------------
    F1A = MSA(DF1, 4)
    o = ConvRelu(64, 1)(F1A)
    o = local_16_C_S_attention(o)
    o = (UpSampling2D((4, 4), data_format=IMAGE_ORDERING, interpolation='bilinear'))(o)
    o = Conv2D(n_classes, 1, activation="softmax")(o)
    return o

