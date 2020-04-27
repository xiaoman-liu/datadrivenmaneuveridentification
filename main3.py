import os
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'
from TypeMapping import lateral_distribution
import pandas as pd
import random
random.seed(0)
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from ImportData import sample_num
from collections import Counter
from utils.model import build_model, calculate_class_weight
# from utils.load_data2 import load_scenario_data, split_trainval
from utils.load_data_9feature_weigted_nonorma import load_scenario_data, split_trainval




if __name__ == "__main__":
    epochs = 150
    model_structure = "two_lstm"
    train_model_name = '{}/%s_%ssample_only_9_feature_weighted12_%sepoch_model{}' % (model_structure, sample_num, epochs)

    # init logging
    # setup_sample_logging(train_model_name)

    # load features and labels
    sample_number, features, lateral_label_onehot, lateral_label_noonehot, masks, _ = load_scenario_data()

    # split train and val sample
    # first split_ratio sample for train, last for val
    split_ratio = 0.8
    split_sample_num = int(sample_number * split_ratio)
    train_features, train_label, val_features, val_label = \
        split_trainval(features, lateral_label_onehot, split_sample_num)

    # print train sample and test sample distribution in each classes
    train_sample_distribution = Counter(lateral_label_noonehot[:split_sample_num].reshape(-1).tolist())
    val_sample_distribution  = Counter(lateral_label_noonehot[split_sample_num:].reshape(-1).tolist())
    print("\nlabel appear in train:\n {}".format(train_sample_distribution))
    print("label appear in val:\n {}".format(val_sample_distribution))
    print("\nlabel appear in train:")
    for key, value in train_sample_distribution.items():
        if key!=-1:
            print("{:15s} {}".format(lateral_distribution[key],value))
    print("\nlabel appear in val:")
    for key, value in val_sample_distribution.items():
        if key !=-1:
            print("{:15s} {}".format(lateral_distribution[key],value))

    # build model
    model = build_model(features)
    model.load_weights("/home/xinjie/xiaoman/codes/datadrivenmaneuveridentification-Xiaoman/weights/two_lstm_150_0.405_checkpoint.hdf5")

    # # load model continue training
    # filename = 'weights/two_lstm_433sample_7feauture_normalized_goon_200epoch_model.h5'
    # model = load_model(filename)
    # model = build_model_cuda(features)

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
                                  factor=0.7937, cooldown=0, min_lr=1e-4, verbose=2)
    model_checkpoint = ModelCheckpoint(filepath='weights/%s_{epoch:d}_{val_loss:.3f}_checkpoint.hdf5'%model_structure, verbose=1,
                                       monitor='val_loss', save_best_only=True, save_weights_only=False,period = 50)
    callback_list = [model_checkpoint, reduce_lr]

    ###### train model
    # calculate classes weight todo:class weight in loss,sample weight
    class_weight = calculate_class_weight(lateral_label_noonehot)
    print("class_weight:\n",class_weight)
    hist = model.fit(train_features, train_label, batch_size=64, epochs=epochs, callbacks=callback_list,#sample_weight =class_weight,
                      verbose=1, validation_data=(val_features, val_label))
    # TODO: use fit generator
    # TODO: try cudnn LSTM



    # save trainlog and model
    log = pd.DataFrame(hist.history)
    log.to_csv(train_model_name.format('trainlog', '.csv'))
    model.save(train_model_name.format('weights', '.h5'))
    # # 保存模型图
    # from keras.utils import plot_model
    #
    # # 需要安装pip install pydot
    # model_plot = '{}/{}_{}_{}_v2.png'.format(params['model_dir'], params['filters'], params['pool_size_1'],
    #                                          params['pool_size_2'])
    # plot_model(model, to_file=model_plot)
