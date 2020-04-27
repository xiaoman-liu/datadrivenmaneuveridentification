import numpy as np
from TypeMapping import lateral_id_mapping,longitudinal_id_mapping
import keras.utils as k_utils
import random
random.seed(0)
from ImportData import sample_num, scenarios_data, top_k_scenario
import os


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


def load_scenario_data():
    ##----------------- scenario_data from importData.py
    # load features and labels
    sample_number = sample_num
    max_timestep = 3500
    # TODO: car type in feature?
    # TODO: feature engineering
    features = np.zeros((sample_number, max_timestep, 10+11+16), dtype=np.float32)
    labels   = np.zeros((sample_number, max_timestep, 8+6), dtype=np.int64)
    labels_cls = -1 * np.ones((sample_number, max_timestep, 2), dtype=np.int64)
    masks    = np.zeros((sample_number, max_timestep), dtype=np.int64)
    index    = 0
    sample_type = []
    random.shuffle(scenarios_data)
    for i, scenario_data in enumerate(scenarios_data):
        if i == top_k_scenario: break
        for key in scenario_data:
            sample_type.append(key)
            type = scenario_data[key]["Type"]
            signal_value = scenario_data[key]['signals'].values[:, list(range(0,7))+list(range(10,13))]

            laneid = change_id_times(scenario_data[key]['signals'].values[:,8])#TODO: laneid without changeinto times
            roadid = change_id_times(scenario_data[key]['signals'].values[:,7])
            onehot_laneid = k_utils.to_categorical(laneid, num_classes=16)
            onehot_roadid = k_utils.to_categorical(roadid, num_classes=11)
            maneuver_temp = scenario_data[key]['maneuverIdentification'].values[:,1:3]
            for idx, item in enumerate(maneuver_temp):
                lateral = lateral_id_mapping[item[0]]
                longtudinal = longitudinal_id_mapping[item[1]]
                lateral_label = k_utils.to_categorical(lateral, num_classes=8)
                longitudinal_label = k_utils.to_categorical(longtudinal, num_classes=6)
                labels[index, idx, :8] = lateral_label
                labels[index, idx, 8:] = longitudinal_label
                labels_cls[index, idx, :] = [lateral, longtudinal]
            features[index, :signal_value.shape[0],:signal_value.shape[1]] = signal_value
            features[index, :signal_value.shape[0], signal_value.shape[1] :signal_value.shape[1] + 11] = onehot_roadid# shape, (num_sample, max_timestep, 12)
            features[index, :signal_value.shape[0], signal_value.shape[1] + 11:signal_value.shape[1] + 27] = onehot_laneid
            masks[index, :signal_value.shape[0]] = 1 # shape, (num_sample, max_timestep)
            index += 1

    #lateral_label_onehot shape = (samples, timesteps,8) # 8 is labels number,after onehot code
    #lateral_label_noonehot shape = (samples, timesteps) label is from 0-7,without onehot
    lateral_label_onehot = labels[:, :, :8]
    lateral_label_noonehot = labels_cls[:, :, 0]

    return sample_number, features, lateral_label_onehot, lateral_label_noonehot, masks, sample_type




def split_trainval(features, lateral_label_onehot,split_num):
    """split train and val sample
    """
    train_features = features[:split_num, :, :]
    train_label = lateral_label_onehot[:split_num,:,:]
    val_features = features[split_num:, :, :]
    val_label = lateral_label_onehot[split_num:,:,:]

    return train_features, train_label, val_features, val_label