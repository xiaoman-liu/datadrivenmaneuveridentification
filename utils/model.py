from keras.layers import Masking, TimeDistributed, GlobalAveragePooling1D, Concatenate, Permute, LSTM, Dense, Dropout, CuDNNLSTM, Bidirectional,Conv1D,BatchNormalization,Activation,Input,add
from keras.models import Sequential,Model
from keras.optimizers import Adam
import math
import random
random.seed(0)
from collections import Counter
import keras.backend as K
import tensorflow as tf
import numpy as np



from keras.engine.topology import Layer


class Self_Attention(Layer):
    """
    Attention layer for RNN models
    """

    def __init__(self, output_dim, return_attention = False,**kwargs):
        self.supports_masking = True
        self.return_attention = return_attention
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight for this layer
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Self_Attention, self).build(input_shape)  # Be sure to call it at the end

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask = None):

        # if mask is not None:
        #     mask = K.cast(mask[..., None], K.floatx())
        #     print("mask.shape", mask.shape)
        #     x *= mask

        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        print("WQ.shape", WQ.shape)
        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)
        print("WV.shape", WV.shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64 ** 0.5)

        if mask is not None:
            mask = K.cast(mask[..., None], K.floatx())
            print("mask.shape", mask.shape)
            QK *= mask
        QK = K.softmax(QK)
        print("QK.shape", QK.shape)
        V = K.batch_dot(QK, WV)
        print("V.shape", V.shape)
        # if self.return_attention:
        #     return [V,]
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)




def weighted_loss(weights):

    def categorical_crossentropy_masked(y_true, y_pred):
        """
        y_true (sample_number, max_timestep, onehot)
        """

        label_weight = tf.convert_to_tensor(weights)
        label_weight = tf.cast(label_weight, tf.float32)
        y_true_weight = label_weight * y_true


        mask = tf.reduce_sum(y_true, axis=-1)  # (sample_num, max_timestep)
        mask = tf.cast(mask, tf.bool)

        y_true_weight = tf.boolean_mask(y_true_weight, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        crossentroy = K.mean(-tf.reduce_sum(y_true_weight * tf.log(y_pred),axis = -1))
        # loss = K.mean(K.categorical_crossentropy(y_true, y_pred))

        return crossentroy

    return categorical_crossentropy_masked



def masked_accuracy(y_true, y_pred):

    mask = tf.reduce_sum(y_true, axis=-1)  # (sample_num, max_timestep)
    mask = tf.cast(mask, tf.bool)

    y_true = tf.boolean_mask(y_true, mask) # (allsamples_left_timesteps, feature_num)
    y_pred = tf.boolean_mask(y_pred, mask)

    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)

    correct_bool = tf.equal(y_true, y_pred)
    mask_accuracy = K.sum(tf.cast(correct_bool, tf.int32)) / tf.shape(correct_bool)[0]
    mask_accuracy = tf.cast(mask_accuracy, tf.float32)

    return mask_accuracy


def calculate_class_weight(lateral_label_noonehot,mu = 0.15):
    """
    calculate class_weight
    """
    label_dict = {}
    all_sample_distribution = Counter(lateral_label_noonehot.reshape(-1).tolist())
    all_sample_distribution = sorted(all_sample_distribution.items(), key=lambda ele: ele[0])
    for key, value in all_sample_distribution:
        if key!=-3:
            label_dict[key] = value
    total = 0
    for _, value in label_dict.items():
        total+=value
    keys = label_dict.keys()
    res = dict()
    res_list = []
    for key in keys:
        score = math.log(mu*total/float(label_dict[key]))
        res[key] = score if score > 1.0 else 1.0
        res_list.append(score if score > 1.0 else 1.0)

    return res,res_list


def build_BLSTM_model(features,weights = np.array(
        [[[1.0, 1.0, 1.2895909855208851,1.8544334641414668]]]),class_number = 4):



    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=features.shape[1:]))
    model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=features.shape[1:])))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    # model.add(Self_Attention(64))
    model.add(TimeDistributed(Dense(class_number, activation="softmax")))
    optm = Adam(lr=0.0001)
    # model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=["accuracy"])
    model.compile(optimizer=optm, loss=weighted_loss(weights), metrics=[masked_accuracy])
    model.summary()

    return model


def build_att_BLSTM_model(features,weights = np.array(
        [[[1.0, 1.0, 1.2895909855208851,1.8544334641414668]]]),class_number = 4):


    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=features.shape[1:]))
    model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=features.shape[1:])))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Self_Attention(64))
    model.add(TimeDistributed(Dense(class_number, activation="softmax")))
    optm = Adam(lr=0.0001)
    # model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=["accuracy"])
    model.compile(optimizer=optm, loss=weighted_loss(weights), metrics=[masked_accuracy])
    model.summary()

    return model


def build_lstm_model(features,weights = np.array(
        [[[1.0, 1.0, 1.2895909855208851,1.8544334641414668]]]),class_number = 4):
    # two layer lstm model
    # Masking layer input shape = (timesteps,features)

    # weights = [[[0,1,10,0,0,0,0,0]]]
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=features.shape[1:]))
    model.add(LSTM(128, return_sequences=True, input_shape=features.shape[1:]))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32,return_sequences = True))
    # model.add(Dense(8, activation="softmax"))
    model.add(TimeDistributed(Dense(class_number, activation="softmax")))
    optm = Adam(lr=0.0001)
    # model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=["accuracy"])
    model.compile(optimizer=optm, loss=weighted_loss(weights), metrics=[masked_accuracy])
    model.summary()

    return model


def build_Attllstm_model(features,weights = np.array(
        [[[1.0, 1.0, 1.2895909855208851,1.8544334641414668]]]),class_number = 4):

    # Masking layer input shape = (timesteps,features)
    # LSTM layer inputshape = (timesteps,features)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=features.shape[1:]))
    model.add(LSTM(128, return_sequences=True, input_shape=features.shape[1:]))
    model.add(LSTM(64, return_sequences=True))
    model.add(Self_Attention(64))
    model.add(TimeDistributed(Dense(class_number, activation="softmax")))
    optm = Adam(lr=0.0001)
    # model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=["accuracy"])
    model.compile(optimizer=optm, loss=weighted_loss(weights), metrics=[masked_accuracy])
    model.summary()

    return model


def build_fcn_model(features,weights = np.array(
        [[[1.0, 1.0, 1.2895909855208851,1.8544334641414668]]]),class_number = 4):
    # weights = [[[0,1,10,0,0,0,0,0]]]
    model = Sequential()
    # model.add(Masking(mask_value=0, input_shape=features.shape[1:]))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', input_shape=features.shape[1:])) #1
    model.add(BatchNormalization())
    model.add(Activation(activation= "relu"))

    model.add(Conv1D(filters=256, kernel_size=3, padding='same'))#2
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv1D(filters=256, kernel_size=3, padding='same'))#3
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv1D(filters=256, kernel_size=3, padding='same'))#4
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv1D(filters=256, kernel_size=3, padding='same'))#5
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv1D(filters=256, kernel_size=5, padding='same'))#6
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv1D(filters=256, kernel_size=5, padding='same'))#7
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv1D(filters=256, kernel_size=5, padding='same'))#8
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv1D(filters=128, kernel_size=7, padding='same'))#9
    model.add(BatchNormalization())
    model.add(Activation("relu"))




    # model.add(GlobalAveragePooling1D())
    model.add(Dense(class_number, activation="softmax"))


    optm = Adam(lr=0.0001)
    # model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=["accuracy"])
    model.compile(optimizer=optm, loss=weighted_loss(weights), metrics=[masked_accuracy])
    model.summary()

    return model


def build_lstm_fcn_model(features, weights  ,class_number = 4):
    ### features.shape = (579,777,17)
    ip = Input(features.shape[1:])

    x = Masking(mask_value=0, input_shape=features.shape[1:])(ip)
    x = LSTM(128, return_sequences= True, input_shape= features.shape[1:])(x)
    x = Dropout(0.2)(x) # x.shape = (?,?,128)

    # y = Permute((2, 1))(ip) # y.shape = (?,17,777)
    y = ip
    y = Conv1D(filters = 128, kernel_size = 8, padding='same')(y) # y.shape = (?,17,128)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(filters = 256, kernel_size = 5, padding='same')(y) # y.shape = (?,17,256)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(filters = 512, kernel_size = 3, padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

     #y.shape = (?,128)

    x = Concatenate(axis = 2)([x, y])

    out = Dense(class_number, activation='softmax')(x)

    model = Model(ip, out)
    optm = Adam(lr=0.0001)
    model.compile(optimizer=optm, loss=weighted_loss(weights), metrics=[masked_accuracy])
    model.summary()

    return model


def build_resnet_model(features,weights = np.array(
        [[[1.0, 1.0, 1.2895909855208851,1.8544334641414668]]]),class_number = 4):

    n_feature_maps = 64

    input_layer = Input(features.shape[1:])

    # BLOCK 1
    conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = add([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)

    # BLOCK 2
    conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_2 = add([shortcut_y, conv_z])
    output_block_2 = Activation('relu')(output_block_2)

    # BLOCK 3
    conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = BatchNormalization()(output_block_2)

    output_block_3 = add([shortcut_y, conv_z])
    output_block_3 = Activation('relu')(output_block_3)

    # FINAL

    # gap_layer = GlobalAveragePooling1D()(output_block_3)

    output_layer = TimeDistributed(Dense(class_number, activation='softmax'))(output_block_3)

    model = Model(inputs=input_layer, outputs=output_layer)
    optm = Adam(lr=0.0001)
    model.compile(optimizer=optm, loss=weighted_loss(weights), metrics=[masked_accuracy])
    # model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=["accuracy"])
    model.summary()

    return model


def build_longtudinal_model(features):
    # two layer lstm model
    # Masking layer input shape = (timesteps,features)
    # LSTM layer inputshape = (timesteps,features)
    weights = np.array(
        [[[0.0, 1.0, 1.8660868401011104, 2.0167500532225158, 1.7543857887550247, 1.0]]])
    # {1: 1.0, 2: 1.8660868401011104, 3: 2.0167500532225158, 4: 1.7543857887550247, 5: 1.0}
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=features.shape[1:]))
    model.add(LSTM(128, return_sequences=True, input_shape=features.shape[1:]))
    model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(1,return_sequences = True))
    # model.add(Dense(8, activation="softmax"))
    model.add(TimeDistributed(Dense(6, activation="softmax")))
    optm = Adam(lr=0.0001)
    # model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=["accuracy"])
    model.compile(optimizer=optm, loss=weighted_loss(weights), metrics=['accuracy'])
    model.summary()

    return model


def build_model_cuda(features):
    # two layer lstm model
    # Masking layer input shape = (timesteps,features)
    # LSTM layer inputshape = (timesteps,features)
    model = Sequential()
    model.add(Masking(mask_value= 0, input_shape=features.shape[1:]))
    model.add(CuDNNLSTM(128, return_sequences = True, input_shape=features.shape[1:]))
    model.add(CuDNNLSTM(64, return_sequences = True))
    # model.add(LSTM(1,return_sequences = True))
    # model.add(Dense(8, activation="softmax"))
    model.add(TimeDistributed(Dense(8, activation="softmax")))
    optm = Adam(lr=0.0001)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=["accuracy"],sample_weight_mode='temporal')
    # model.compile(optimizer=optm, loss=categorical_crossentropy_masked, metrics=['accuracy'])
    model.summary()

    return model