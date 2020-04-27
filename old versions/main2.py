import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'
import numpy as np
from TypeMapping import *
import keras.backend as K
import keras.utils as k_utils
import pandas as pd
import tensorflow as tf
import random
random.seed(0)
from keras.layers import Masking, TimeDistributed, LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from ImportData import *
from collections import Counter


# TODO: check mask in loss and Masking Layer
def categorical_crossentropy_masked(y_true, y_pred):
    """
    y_true (sample_number, max_timestep, onehot)
    """
    # mask = y_true[:, :, 0:1] != 1
    # mask = tf.dtypes.cast(mask, tf.float32)
    # y_true = y_true * mask
    # y_pred = y_pred * mask
    mask = y_true[:, :, 0] != 1
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    loss = K.mean(K.categorical_crossentropy(y_true, y_pred))
    return loss

# build model compile
def build_model(features):
    model = Sequential()
    model.add(Masking(mask_value= 0, input_shape=features.shape[1:]))
    model.add(LSTM(128, return_sequences = True, input_shape=features.shape[1:]))
    model.add(LSTM(64, return_sequences = True))
    # model.add(LSTM(1,return_sequences = True))
    # model.add(Dense(6, activation="softmax"))
    model.add(TimeDistributed(Dense(9, activation="softmax")))
    optm = Adam(lr=0.0001)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=optm, loss=categorical_crossentropy_masked, metrics=['accuracy'])

    return model


if __name__ == "__main__":
    ##----------------- scenario_data from importData.py
    sample_number = sample_num
    max_timestep = 3500
    # TODO: car type in feature?
    # TODO: check one hot.
    # TODO: feature engineering
    features = np.zeros((sample_number, max_timestep, 12), dtype=np.float32)
    labels   = np.zeros((sample_number, max_timestep, 2), dtype=np.int64)
    masks    = np.zeros((sample_number, max_timestep), dtype=np.int64)
    index    = 0
    random.shuffle(scenarios_data)
    for i, scenario_data in enumerate(scenarios_data):
        if i == top_k_scenario: break
        for key in scenario_data:
            signal_value = scenario_data[key]['signals'].values[:, list(range(0,9))+list(range(10,13))]
            maneuver_temp = scenario_data[key]['maneuverIdentification'].values[:,1:3]
            for idx, item in enumerate(maneuver_temp):
                lateral = lateral_id_mapping[item[0]]
                longtudinal = longitudinal_id_mapping[item[1]]
                labels[index, idx, :] = [lateral, longtudinal]


            features[index,:len(signal_value),:] = signal_value # shape, (num_sample, max_timestep, 12)
            masks[index, :len(signal_value)] = 1 # shape, (num_sample, max_timestep)
            index += 1

    lateral_label_noonehot = labels[:, :, 0]
    lateral_label = k_utils.to_categorical(lateral_label_noonehot, num_classes=9)

    model = build_model(features)

    model.summary()
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
                                  factor=0.7937, cooldown=0, min_lr=1e-4, verbose=2)

    model_checkpoint = ModelCheckpoint("weights/weights.h5", verbose=1,
                                       monitor='valloss', save_best_only=True, save_weights_only=True)

    callback_list = [model_checkpoint, reduce_lr]

    epochs = 500

    # split train and val sample
    split_ratio = 0.8 # first split_ratio sample for train, last for val
    split_sample_num = int(sample_number * split_ratio)
    train_features = features[:split_sample_num, :, :]
    train_label = lateral_label[:split_sample_num,:,:]
    val_features = features[split_sample_num:, :, :]
    val_label = lateral_label[split_sample_num:,:,:]


    train_sample_distribution = Counter(lateral_label_noonehot[:split_sample_num].reshape(-1).tolist())
    test_sample_distribution  = Counter(lateral_label_noonehot[split_sample_num:].reshape(-1).tolist())
    print("\nlabel appear in train:\n {}".format(train_sample_distribution))
    print("label appear in val:\n {}".format(test_sample_distribution))
    print("\nlabel appear in train:")
    for key, value in train_sample_distribution.items():
        if key: print("{:15s} {}".format(lateral_distribution[key],value))

    print("\nlabel appear in val:")
    for key, value in test_sample_distribution.items():
        if key: print("{:15s} {}".format(lateral_distribution[key],value))


    # TODO: check accuracy in val
    hist = model.fit(train_features, train_label, batch_size=64, epochs=epochs, callbacks=callback_list,
                      verbose=1, validation_data=(val_features, val_label))
    log = pd.DataFrame(hist.history)
    log.to_csv('train_log.csv')
    model.save('weights/model_last.h5')