from keras.layers import Masking, TimeDistributed, LSTM, Dense, CuDNNLSTM
from keras.models import Sequential
from keras.optimizers import Adam
import math
import random
random.seed(0)
from collections import Counter
import keras.backend as K
import tensorflow as tf
import numpy as np



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
        if key!=-1:
            label_dict[key] = value
    total = 0
    for _, value in label_dict.items():
        total+=value
    keys = label_dict.keys()
    res = dict()
    for key in keys:
        score = math.log(mu*total/float(label_dict[key]))
        res[key] = score if score > 1.0 else 1.0

    return res


def build_model(features):
    # two layer lstm model
    # Masking layer input shape = (timesteps,features)
    # LSTM layer inputshape = (timesteps,features)
    weights = np.array(
        [[[1.6413955164256195, 1.0, 1.0, 1.0, 1.0, 3.0370399682846183, 1.7239681905050477, 2.675769539175348]]])
    # weights = [[[0,1,10,0,0,0,0,0]]]
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=features.shape[1:]))
    model.add(LSTM(128, return_sequences=True, input_shape=features.shape[1:]))
    model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(1,return_sequences = True))
    # model.add(Dense(8, activation="softmax"))
    model.add(TimeDistributed(Dense(8, activation="softmax")))
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