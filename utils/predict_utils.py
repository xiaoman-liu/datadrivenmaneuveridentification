from keras.optimizers import Adam
import numpy as np
import pandas as pd
from TypeMapping import lateral_distribution
import keras.backend as K
import tensorflow as tf
import random
random.seed(0)
import matplotlib
matplotlib.use('Agg')
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.layers import Masking, TimeDistributed, LSTM, Dense, CuDNNLSTM
from keras.models import Sequential
from keras.optimizers import Adam
import os

import math
import random
from sklearn.metrics import confusion_matrix
# from prediction3 import model_name

# class_weight = {0: 1.8681419262260441, 1: 1.0, 2: 1.0, 4: 1.0, 5: 2.6624168700615383, 6: 1.9883759054811836, 7: 2.9052259214862484}
# weight = {
#     0: 1.6413955164256195, 1: 1.0, 2: 1.0, 4: 1.0, 5: 3.0370399682846183, 6: 1.7239681905050477, 7: 2.675769539175348
# }

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


def build_model(features):
    # two layer lstm model
    # Masking layer input shape = (timesteps,features)
    # LSTM layer inputshape = (timesteps,features)
    weights = np.array([[[1.6413955164256195, 1.0, 1.0, 1.0,1.0, 3.0370399682846183, 1.7239681905050477, 2.675769539175348]]])

    model = Sequential()
    model.add(Masking(mask_value= 0, input_shape=features.shape[1:]))
    model.add(LSTM(128, return_sequences = True, input_shape=features.shape[1:]))
    model.add(LSTM(64, return_sequences = True))
    # model.add(LSTM(1,return_sequences = True))
    # model.add(Dense(8, activation="softmax"))
    model.add(TimeDistributed(Dense(8, activation="softmax")))
    optm = Adam(lr=0.0001)
    # model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=["accuracy"],sample_weight_mode='temporal')
    model.compile(optimizer=optm, loss=weighted_loss(weights), metrics=['accuracy'])
    model.summary()

    return model

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


def get_filePath_fileName_fileExt(filename):
    """get file path ,file name,file extension"""

    (filepath,tempfilename) = os.path.split(filename)
    (shotname,extension) = os.path.splitext(tempfilename)

    return filepath,shotname,extension


def classifaction_report_csv(report,model_name,scenario_keys,savepath):
    """ save classifaction report as csv
     include total sample number,precision,recall,f1_score in each class"""
    df = pd.DataFrame(report).transpose()
    df.to_csv(savepath + "/plots/%s_%s_classification_report.csv"%(model_name,scenario_keys),index = True)



def plot_confusion_matrix(cm, cf_label,model_name,scenario_keys,savepath):
    """
    plot confusion matrix
    """
    savename = savepath + "/plots/%s_%s_cm_matrix.png" % (model_name, scenario_keys)
    # savename = 'test_log/%s_%s_cm_matrix.png' % (model_name, scenario_keys)
    title = '%s_Confusion_Matrix' % (model_name)
    #plot confusion matrix
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=3)

    # Probability value of each cell in the confusion matrix
    ind_array = np.arange(len(cf_label))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%d" % (c,), color='tomato', fontsize=13, va='center', ha='center')
            # cm_normalized plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.BuGn)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(cf_label)))
    plt.xticks(xlocations, cf_label, rotation=90)
    plt.yticks(xlocations, cf_label)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(cf_label))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.close()
    # plt.show()


def plot_sample(sample_number, val_pred_cls, val_true_cls, masks, sample_type,model_name,scenario_keys,savepath,scenarios_name):
    """plotting sample unit figure, and calculate accuracy, precision,recall,f1score,cm for each sample"""

    for index in range(sample_number):
        val_pred_cls_sample = val_pred_cls[index]
        val_true_cls_sample = val_true_cls[index]
        mask_sample = masks[index].astype(np.bool)
        val_pred_cls_sample = val_pred_cls_sample[mask_sample]
        val_true_cls_sample = val_true_cls_sample[mask_sample]
        correct_bool = val_pred_cls_sample == val_true_cls_sample
        accuracy = correct_bool.astype(np.int).sum() / len(correct_bool)
        num_label = list(set(list(np.unique(val_pred_cls_sample))+list(np.unique(val_true_cls_sample))))
        str_label = [lateral_distribution[i] for i in num_label]

        ## calculate  precision,recall,f1score,cm for each sample
        # cm = confusion_matrix(val_true_cls_sample, val_pred_cls_sample)
        # print("confusion_matrix,without normalization\n", cm)
        # plt.matshow(cm, cmap=plt.cm.gray)
        # plt.show()
        # ave_precision = precision_score(val_true_cls_sample, val_pred_cls_sample, average='macro')
        # print("average_precision_score\n", ave_precision)
        # ave_recall = recall_score(val_true_cls_sample, val_pred_cls_sample, average='macro')
        # print("average_recall_score\n", ave_recall)
        # ave_fiscore = f1_score(val_true_cls_sample, val_pred_cls_sample, average='macro')
        # print("average_f1_score\n", ave_fiscore)

        plt.plot(val_pred_cls_sample, 'g-.', label="predict")
        plt.plot(val_true_cls_sample, 'r--', label="gt")
        plt.legend()
        plt.xlabel("timesteps")
        # plt.ylabel("maneuver")
        plt.yticks(num_label,str_label,fontsize = 10,rotation = 60)
        plt.title("{} sample_{}_ accuracy {:.4f}".format(sample_type[index], scenarios_name[index],accuracy))
        save_name =  savepath + "/plots/{}_%s_sample{}_%s" % (scenarios_name[index], model_name)

        plt.savefig(save_name.format(sample_type[index],index))
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


    ## output prediction result in txt file
    # out_path = "test_log/{}_predictions.txt".format(model_name)
    # file = open(out_path, "w")
    # file.write("\ntest pred_cls index\n")
    # file.write(", ".join(map(str, val_pred_cls)))
    # file.write("\ntest true_cls index\n")
    # file.write(", ".join(map(str, val_true_cls)))
    # file.write("\ncorrect_bool\n")
    # file.write(", ".join(map(str, correct_bool)))
    # file.write("\ntest accuracy: {}".format(accuracy))
    # file.close()