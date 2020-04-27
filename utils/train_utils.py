import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'
import keras.backend as K
import tensorflow as tf
import random
random.seed(0)
from sklearn.metrics import confusion_matrix,precision_score
import logging
import time
import math






def DTWDistance(s1, s2):
    # Calculate the similarity of samples
    # s1,s2 are sample 1,sample 2
    # sample 1 inputshape = (timesteps,)

    DTW={}
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return math.sqrt(DTW[len(s1)-1, len(s2)-1])



def ave_pre(y_true,y_pred):

    mask = tf.reduce_sum(y_true, axis=-1)  # (sample_num, max_timestep)
    mask = tf.cast(mask, tf.bool)

    y_true = tf.boolean_mask(y_true, mask) # (allsamples_left_timesteps, feature_num)
    y_pred = tf.boolean_mask(y_pred, mask)

    # y_true = K.argmax(y_true, axis=-1)
    # y_pred = K.argmax(y_pred, axis=-1)
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # y_pred = tf.cast(y_pred, tf.float32)
    # y_true = tf.cast(y_true, tf.float32)
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # precision = true_positives / (predicted_positives + K.epsilon())

    # precision = y_pred / (predicted_positives + K.epsilon())


    val_true_cls = K.eval(y_true)
    val_pred_cls = K.eval(y_pred)
    # sess.run(tf.global_variables_initializer())
    # val_true_cls = y_true.eval(session=sess)
    # val_pred_cls = y_pred.eval(session=sess)

    precision = precision_score(val_true_cls, val_pred_cls, average='macro')

    precision = tf.cast(precision, tf.float32)

    return precision


def ave_recall(y_true, y_pred):

    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())

    return recall

def ave_f1(y_true, y_pred):

    precision = ave_pre(y_true, y_pred)
    recall = ave_recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def confusion(y_true, y_pred):

    mask = tf.reduce_sum(y_true, axis=-1)  # (sample_num, max_timestep)
    mask = tf.cast(mask, tf.bool)

    y_true = tf.boolean_mask(y_true, mask) # (allsamples_left_timesteps, feature_num)
    y_pred = tf.boolean_mask(y_pred, mask)

    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.eval(y_true).reshape(-1)
    y_pred = K.eval(y_pred).reshape(-1)
    cm = confusion_matrix(y_true, y_pred)

    return cm


def setup_sample_logging(train_model_name):
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    LOG_FORMAT = "%Y-%m-%d_%H-%M"
    LOG_TIME = time.strftime(LOG_FORMAT, time.localtime(time.time()))
    LOG_TRAIN_NAME = os.path.join(train_model_name.format('trainlog/samplelogs', LOG_TIME))
    logging.basicConfig(filename=LOG_TRAIN_NAME, format='%(asctime)s\t%(message)s',
                        level=logging.INFO, datefmt=LOG_FORMAT, filemode='w')
