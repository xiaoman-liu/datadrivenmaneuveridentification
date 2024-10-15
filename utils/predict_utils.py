from keras.optimizers import Adam
import numpy as np
import pandas as pd
from TypeMapping import lanemap,vehiclestate_map
import keras.backend as K
import tensorflow as tf
import random
random.seed(0)
import matplotlib
matplotlib.use('Agg')
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.layers import Masking, TimeDistributed, LSTM, Dense, CuDNNLSTM,Bidirectional,Conv1D,BatchNormalization,Activation,add,Input
from keras.models import Sequential,Model
from keras.optimizers import Adam
import os
from keras.engine.topology import Layer
import math
import random
from sklearn.metrics import confusion_matrix
from collections import Counter



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

        crossentroy = -tf.reduce_sum(y_true_weight * tf.log(y_pred),axis = -1)
        # loss = K.mean(K.categorical_crossentropy(y_true, y_pred))

        return K.mean(crossentroy)
    return categorical_crossentropy_masked



def evaluate_model(modelname,features,lateral_label):
    # evaluate model

    _,__,Extension = get_filePath_fileName_fileExt(modelname)

    if Extension  == ".hdf5":
        model  = build_model(features)

        model.load_weights(modelname)
        loss, accuracy = model.evaluate(features, lateral_label)
    elif Extension == ".h5":
        model = load_model(modelname)
        loss, accuracy = model.evaluate(features, lateral_label)
    else:

        return "Not effective modelname"

    return loss,accuracy

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

    return res_list


def change_id_times(feature_id):
    # change the roadid into the times of changeroad
    # featureid shape = ()

    a = feature_id[0]
    change_time = 0
    for i in range(len(feature_id)):
        if feature_id[i] != a:
            change_time += 1
            a = feature_id[i]
        feature_id[i] = change_time

    return feature_id

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


def get_filePath_fileName_fileExt(filename):
    """get file path ,file name,file extension"""

    (filepath,tempfilename) = os.path.split(filename)
    (shotname,extension) = os.path.splitext(tempfilename)

    return filepath,shotname,extension


def classifaction_report_csv(report,model_name,savepath):
    """ save classifaction report as csv
     include total sample number,precision,recall,f1_score in each class"""
    df = pd.DataFrame(report).transpose()
    df.to_csv(savepath + "/%s_classification_report.csv"%(model_name),index = True)



def plot_confusion_matrix(cm, cf_label,model_name,savepath):
    """
    plot confusion matrix
    """
    savename = savepath + "/%s_cm_matrix.png" % (model_name)
    # savename = 'test_log/%s_%s_cm_matrix.png' % (model_name, scenario_keys)
    title = '%s_Confusion_Matrix' % (model_name)
    #plot confusion matrix

    plt.rcParams['savefig.dpi'] = 600
    plt.figure(figsize=(9, 8), dpi=600)
    np.set_printoptions(precision=3)

    # Probability value of each cell in the confusion matrix
    ind_array = np.arange(len(cf_label))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%d" % (c,), color='tomato', fontsize=20, va='center', ha='center')
            # cm_normalized plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.BuGn)
    # plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(cf_label)))
    plt.xticks(xlocations, cf_label,fontsize = 16,rotation = 15)
    plt.yticks(xlocations, cf_label,fontsize = 16)
    plt.ylabel('Actual label',fontsize = 16)
    plt.xlabel('Predict label',fontsize = 16)

    # offset the tick
    tick_marks = np.array(range(len(cf_label))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.tight_layout()
    plt.savefig(savename, format='png')
    plt.close()
    # plt.show()


def plot_sample(sample_number, val_pred_cls, val_true_cls, masks,model_name,savepath,maneuver_map = None):
    """plotting sample unit figure, and calculate accuracy, precision,recall,f1score,cm for each sample"""

    name = []
    for index in range(sample_number):
        val_pred_cls_sample = val_pred_cls[index]
        val_true_cls_sample = val_true_cls[index]
        mask_sample = masks[index].astype(np.bool)
        val_pred_cls_sample = val_pred_cls_sample[mask_sample]
        val_true_cls_sample = val_true_cls_sample[mask_sample]
        correct_bool = val_pred_cls_sample == val_true_cls_sample
        accuracy = correct_bool.astype(np.int).sum() / len(correct_bool)
        num_label = list(set(list(np.unique(val_pred_cls_sample))+list(np.unique(val_true_cls_sample))))
        str_label = [maneuver_map[i] for i in num_label]


        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['figure.dpi'] = 600

        plt.plot(val_pred_cls_sample, 'g-.', label="predict")
        plt.plot(val_true_cls_sample, 'r--', label="gt")
        plt.legend()
        plt.xlabel("timesteps",fontsize = 11)
        plt.yticks(num_label,str_label,fontsize = 11,rotation = 60)

        save_name =  savepath + "/plots/sample%d_%2d" % (index,accuracy *100)
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close()



        # plt.show()



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
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    # precision = y_pred / (predicted_positives + K.epsilon())
    # val_true_cls = K.eval(y_true)
    # val_pred_cls = K.eval(y_pred)
    # sess.run(tf.global_variables_initializer())
    # val_true_cls = y_true.eval(session=sess)
    # val_pred_cls = y_pred.eval(session=sess)
    # precision = precision_score(val_true_cls, val_pred_cls, average='macro')
    # precision = tf.cast(precision, tf.float32)
    return precision


def confusion(y_true, y_pred):

    mask = tf.reduce_sum(y_true, axis=-1)  # (sample_num, max_timestep)
    mask = tf.cast(mask, tf.bool)



    y_true = tf.boolean_mask(y_true, mask) # (allsamples_left_timesteps, feature_num)
    y_pred = tf.boolean_mask(y_pred, mask)

    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    cm = confusion_matrix(y_true, y_pred)

    return cm

