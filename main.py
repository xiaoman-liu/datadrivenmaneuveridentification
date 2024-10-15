import os
import numpy as np
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'
from TypeMapping import lanemap,vehiclestate_map, turn_map,preceding_map,categorymap
import pandas as pd
import random
random.seed(0)
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from ImportData import sample_num
from collections import Counter
from utils.model import calculate_class_weight,build_lstm_model,build_fcn_model,build_resnet_model,build_BLSTM_model, build_lstm_fcn_model



from utils.load_data import load_data_lane, split_trainval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maneuver identification")
    parser.add_argument("-model", default="LSTM",
                        help="select model for training, FCN, RESNET, LSTM, BLSTM, LSTM-FCN, 5 models are available")
    parser.add_argument("-epoch", default=200, help="training epoch")
    # total four labels: category = ["preceding","turn","lane","vehiclestate"]
    parser.add_argument("-category", default="turn", help="4 maneuver labels, turn, preceding, lane, vehiclestate")
    parser.add_argument("-split_ratio", default=0.8, help="ratio between sample number from train and val")
    args = parser.parse_args()

    epochs = int(args.epoch)
    category = args.category
    model_structure = args.model + '_' + category
    # split train and val sample
    # first split_ratio sample for train, last for val
    split_ratio = args.split_ratio

    train_model_name = '{}/%s_%ssample_%sepoch{}' % (model_structure, sample_num, epochs)

    # load features and labels
    sample_number, features, label_onehot, label_noonehot, masks, change_label =\
        load_data_lane(class_number = categorymap[category],category = category)

    split_sample_num = int(sample_number * split_ratio)
    train_features, train_label, val_features, val_label = \
        split_trainval(features, label_onehot, split_sample_num)

    # print train sample and test sample distribution in each class
    train_sample_distribution = Counter(label_noonehot[:split_sample_num].reshape(-1).tolist())
    val_sample_distribution  = Counter(label_noonehot[split_sample_num:].reshape(-1).tolist())
    print("\nlabel appear in train:\n {}".format(train_sample_distribution))
    print("label appear in val:\n {}".format(val_sample_distribution))
    print("\nlabel appear in train:")
    for key, value in train_sample_distribution.items():
        if key!=-3:
            if category == "preceding":
                print("{:15s} {}".format(preceding_map[key], value))
            elif category == "turn":
                print("{:15s} {}".format(turn_map[key], value))
            elif category == "lane":
                print("{:15s} {}".format(lanemap[key],value))
            else:
                print("{:15s} {}".format(vehiclestate_map[key], value))
    print("\nlabel appear in val:")
    for key, value in val_sample_distribution.items():
        if key !=-3:
            if category == "preceding":
                print("{:15s} {}".format(preceding_map[key], value))
            elif category == "turn":
                print("{:15s} {}".format(turn_map[key], value))
            elif category == "lane":
                print("{:15s} {}".format(lanemap[key],value))
            else:
                print("{:15s} {}".format(vehiclestate_map[key], value))


    # calculate classes weight
    res_dict,class_weight = calculate_class_weight(label_noonehot)
    print("class_weight:\n",res_dict,class_weight)

    # build model
    if "FCN" in model_structure:
        model = build_fcn_model(features,class_number= categorymap[category],weights = np.array(class_weight).reshape((1,1,-1)))
    elif "RESNET" in model_structure:
        model = build_resnet_model(features,class_number= categorymap[category],weights = np.array(class_weight).reshape((1,1,-1)))
    elif "LSTM" in model_structure:
        model = build_lstm_model(features,class_number= categorymap[category],weights = np.array(class_weight).reshape((1,1,-1)))
    elif "Lstm-fcn" in model_structure:
        model = build_lstm_fcn_model(features, class_number=categorymap[category],
                                 weights=np.array(class_weight).reshape((1, 1, -1)))
    else: # "BLSTM"
        model = build_BLSTM_model(features,class_number= categorymap[category],weights = np.array(class_weight).reshape((1,1,-1)))


    ## load model continue training
    # model.load_weights("/home/xinjie/xiaoman/ppt presentation/7-13/lstm/Turn/lstm-turn_724sample_yesid_200epoch.h5")


    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
                                  factor=0.7937, cooldown=0, min_lr=1e-5, verbose=2)
    model_checkpoint = ModelCheckpoint(filepath='weights/%s_{epoch:d}_{val_loss:.3f}_checkpoint.hdf5'%model_structure, verbose=1,
                                       monitor='val_loss', save_best_only=False, save_weights_only=False,period = 50)
    callback_list = [model_checkpoint, reduce_lr]

    # train model
    hist = model.fit(train_features, train_label, batch_size=64, epochs=epochs, callbacks=callback_list,
                      verbose=2, validation_data=(val_features, val_label))

    # save trainlog and model
    log = pd.DataFrame(hist.history)
    log.to_csv(train_model_name.format('trainlog', '.csv'))
    model.save(train_model_name.format('weights', '.h5'))
