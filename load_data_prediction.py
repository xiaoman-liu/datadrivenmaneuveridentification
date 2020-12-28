import numpy as np
from TypeMapping import vehicle_state_id_mapping,lane_id_mapping,turn_id_mapping,preceding_id_mapping
import keras.utils as k_utils
import random
random.seed(0)
from ImportData import sample_num, data_list, max_length
import os
import math


def load_data_lane(prediction = False,class_number = 4,category = "",ratio = 0.8):
    ##----------------- scenario_data from importData.py
    # load features and labels


    sample_number = sample_num
    max_timestep = max_length
    # TODO: car type in feature?
    # TODO: feature engineering,data bucketing
    features    = np.zeros((sample_number, max_timestep, 17), dtype=np.float32)
    labels      = np.zeros((sample_number, max_timestep, class_number), dtype=np.int64)
    labels_cls  = -3 * np.ones((sample_number, max_timestep, 2), dtype=np.int64)
    masks       = np.zeros((sample_number, max_timestep), dtype=np.int64)
    index       = 0



    for i, sample_data in enumerate(data_list):
        # time = sample_data[:,[1]] / np.array([728])
        numerical_feature = np.array(sample_data[:,[1] + list(range(4,11)) + [12,13,25,27,28,29]]) \
                            / np.array([800,1000,1000,10,5,5,10,2,10,1,50,5,1,1],dtype= np.float32)

        laneid = np.array(sample_data[:,[11]])
        if_changelane = np.zeros_like(laneid)
        first_laneid = []
        for id in range(len(laneid)):
            if id == 0:
                first_laneid.append(laneid[0])
            else:
                if laneid[id] not in first_laneid:
                    first_laneid.append(laneid[id])
                if laneid[id] != first_laneid[0]:
                    if_changelane[id] = 1

        if_changelane = k_utils.to_categorical(if_changelane, num_classes=2)

        postion_x = np.array(sample_data[:,[4]], dtype=np.float32)
        position_y = np.array(sample_data[:,[5]], dtype=np.float32)
        angel = np.arctan(position_y / postion_x + 1e-6) * 180 / math.pi / 360


        # maneuver
        if category == "preceding":
            maneuver_temp = sample_data[:,20]
        elif category == "turn":
            maneuver_temp = sample_data[:, 21]
        elif category == "lane":
            maneuver_temp = sample_data[:, 17]
        else:
            maneuver_temp = sample_data[:, 22]

        for idx, item in enumerate(maneuver_temp):
            if category == "preceding":
                maneuver = preceding_id_mapping[item]
            elif category == "turn":
                maneuver = turn_id_mapping[item]
            elif category == "lane":
                maneuver = lane_id_mapping[item]
            else:
                maneuver = vehicle_state_id_mapping[item]

            maneuver_temp[idx] = maneuver
            label = k_utils.to_categorical(maneuver, num_classes=class_number)
            labels[index, idx, :class_number] = label
            labels_cls[index, idx, :] = maneuver

        # add features
        features[index, :numerical_feature.shape[0], :numerical_feature.shape[1]] = numerical_feature
        features[index, :numerical_feature.shape[0], numerical_feature.shape[1]:numerical_feature.shape[1] +1] = angel
        features[index, :numerical_feature.shape[0], numerical_feature.shape[1] +1:numerical_feature.shape[1] + 3] = if_changelane

        masks[index, :sample_data.shape[0]] = 1 # shape, (num_sample, max_timestep)
        index += 1

    lateral_label_onehot = labels[:, :, :class_number]
    lateral_label_noonehot = labels_cls[:, :, 0]

    split = int(sample_number * ratio)
    if prediction == True:
        sample_number = sample_number - split

        return sample_number, features[split:,:,:], lateral_label_onehot[split:,:,:], lateral_label_noonehot[split:,:], masks[split:,:]
    else:

        return split, features[:split,:,:], lateral_label_onehot[:split,:,:], lateral_label_noonehot[:split,:], masks[:split,:]
    # return sample_number, no_pad_features, no_pad_labels, lateral_label_noonehot, masks, sample_type,scenarios_name


def split_trainval(features, lateral_label_onehot,split_num):
    """split train and val sample
    """
    train_features = features[:split_num, :, :]
    train_label = lateral_label_onehot[:split_num,:,:]
    val_features = features[split_num:, :, :]
    val_label = lateral_label_onehot[split_num:,:,:]
    # train_features = features[:split_num]
    # train_label = lateral_label_onehot[:split_num]
    # val_features = features[split_num:]
    # val_label = lateral_label_onehot[split_num:]

    return train_features, train_label, val_features, val_label


def get_filePath_fileName_fileExt(filename):
    """get file path ,file name,file extension"""

    (filepath,tempfilename) = os.path.split(filename)
    (shotname,extension) = os.path.splitext(tempfilename)

    return filepath,shotname,extension


def change_id_times(feature_id):
    """
    change the roadid into the times of changeroad
    featureid shape = ()
    """

    a = feature_id[0]
    change_time = 0
    for i in range(len(feature_id)):
        if feature_id[i] != a:
            change_time += 1
            a = feature_id[i]
        feature_id[i] = change_time

    return feature_id

