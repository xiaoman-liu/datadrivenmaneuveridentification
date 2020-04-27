import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'
import numpy as np
from keras.models import load_model
from TypeMapping import lateral_distribution
from Import_Data_prediction import scenario_keys
import random
from utils.predict_utils import build_model
random.seed(0)
from collections import Counter
from sklearn.metrics import confusion_matrix,classification_report

from utils.load_data_9feature_weigted_nonorma import load_scenario_data, get_filePath_fileName_fileExt
from utils.predict_utils import plot_sample,plot_confusion_matrix,classifaction_report_csv,evaluate_model,weighted_loss



if __name__ == "__main__":
    # load features and labels
    sample_number, features, lateral_label_onehot, lateral_label_noonehot, masks, sample_type, scenarios_name = load_scenario_data()

    #print test sample distribution in each classes
    test_sample_distribution = Counter(lateral_label_noonehot.reshape(-1).tolist())
    test_sample_distribution = sorted(test_sample_distribution.items(), key=lambda ele: ele[0])
    print(test_sample_distribution)
    print("\nlabel appear in test:\n")
    for key, value in test_sample_distribution:
        if key!= -1:
            print("{:15s} {}".format(lateral_distribution[key], value))


    ### load model
    savepath = "/home/xinjie/xiaoman/ppt presentation/4-27/9_features_roadlanechange"
    filename = savepath + "/two_lstm_250_382_checkpoint.hdf5"


    # # load full saved model
    # model = load_model(filename)
    # _, model_name, __ = get_filePath_fileName_fileExt(filename)

    # load model weight
    model = build_model(features)
    model.load_weights(filename)

    _, model_name, _ = get_filePath_fileName_fileExt(filename)

    # evaluate model
    # loss,accuracy = evaluate_model(filename,features,lateral_label_onehot)

    ## mask the ground truth and predicted labels
    sample_index = 0
    val_feature = features[sample_index:, :, :]
    val_lateral_label = lateral_label_noonehot[sample_index:, :]
    val_mask = masks[sample_index:, :].astype(np.bool)
    val_pred = model.predict(val_feature)
    val_pred_cls = np.argmax(val_pred[:,:,:], axis=-1)
    val_true_cls = val_lateral_label



    # plotting sample unit figure, and calculate accuracy, precision,recall,f1score,cm for each sample
    plot_sample(sample_number, val_pred_cls, val_true_cls, masks, sample_type,model_name,scenario_keys,savepath,scenarios_name)

    # mask all predicted labels
    val_pred_cls = val_pred_cls[val_mask]
    val_true_cls = val_true_cls[val_mask]
    correct_bool = val_pred_cls == val_true_cls

    # # calculate  average accuracy, precision,recall,f1score,cm for each class
    # accuracy = correct_bool.astype(np.int).sum() / len(correct_bool)
    # print("accuracy\n", accuracy)
    # ave_precision = precision_score(val_true_cls, val_pred_cls, average=None)
    # print("average_precision_score\n", ave_precision)
    # ave_recall = recall_score(val_true_cls, val_pred_cls, average=None)
    # print("average_recall_score\n", ave_recall)
    # ave_fi_score = f1_score(val_true_cls, val_pred_cls, average=None)
    # print("average_f1_score\n", ave_fi_score)

    ## visualize the confusion matrix for all classes
    cf_label = list(np.unique(val_pred_cls)) + list(np.unique(val_true_cls))
    cf_labels_num = sorted(np.unique(cf_label))
    cf_label_name = []
    for key,value in enumerate(cf_labels_num):
        cf_label_name.append(lateral_distribution[value])
    cm = confusion_matrix(val_true_cls, val_pred_cls)
    ## visualize the normalized confusion matrix for all classes
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("confusion_matrix,without normalization\n", cm)
    plot_confusion_matrix(cm, cf_label_name,model_name,scenario_keys,savepath)



    # print the average classification_report for each class and save as csv
    print(classification_report(val_true_cls, val_pred_cls, digits=4,target_names=cf_label_name))
    report = classification_report(val_true_cls, val_pred_cls, digits=4, target_names=cf_label_name, output_dict=True)
    classifaction_report_csv(report,model_name,scenario_keys,savepath)

