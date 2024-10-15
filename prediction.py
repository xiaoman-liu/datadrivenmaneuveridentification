import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'
from TypeMapping import vehiclestate_map,lanemap,turn_map,preceding_map,categorymap
import random
from utils.predict_utils import plot_sample,plot_confusion_matrix,classifaction_report_csv
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
from ImportData import sample_num
from utils.predict_utils import plot_sample
from collections import Counter
from utils.model import build_lstm_model, calculate_class_weight,build_fcn_model,build_resnet_model,build_BLSTM_model
from utils.load_data_prediction import load_data_lane, split_trainval,get_filePath_fileName_fileExt
random.seed(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Maneuver identification")
    parser.add_argument("-model", default="FCN",
                        help="select model for test, FCN, RESNET, LSTM, BLSTM 4 models are available")
    # total four labels: category = ["preceding","turn","lane","vehiclestate"]
    parser.add_argument("-category", default="turn", help="4 maneuver labels, turn, preceding, lane, vehiclestate")
    parser.add_argument("-split_ratio", default=0.8, help="ratio between sample number from train and val")
    parser.add_argument("-save_path", default="D:/xiaoman/晓曼大大的文件夹/毕业设计/experiments/FCN/turn/9-layer-128-256-128-3-3-3-3-3-5-5-5-7-29/400epoch", help="save path for the prediction results")
    parser.add_argument("-model_path", default="D:/xiaoman/晓曼大大的文件夹/毕业设计/experiments/FCN/turn/9-layer-128-256-128-3-3-3-3-3-5-5-5-7-29/400epoch/", help="the path of model for prediction")
    parser.add_argument("-model_name", default="FCN_turn_200_0142_checkpoint.hdf5", help="the name of model for prediction")
    args = parser.parse_args()

    category = args.category
    model_structure = args.model + '_' + category

    ### load modelhome/xinji
    savepath = args.save_path
    model_select = args.model
    filename = args.model_path + args.model_name


    # split train and val sample
    # first split_ratio sample for train, last for val
    split_ratio = args.split_ratio

    train_model_name = '{}/%s_%ssample_{}' % (model_structure, sample_num)

    # load features and labels
    sample_number, features, label_onehot, label_noonehot, masks= load_data_lane( prediction=True ,class_number=categorymap[category],category = category, ratio = split_ratio)

    # print test sample distribution in each classes
    test_sample_distribution = Counter(label_noonehot.reshape(-1).tolist())
    test_sample_distribution = sorted(test_sample_distribution.items(), key=lambda ele: ele[0])
    print(test_sample_distribution)
    print("\nlabel appear in test:\n")
    for key, value in test_sample_distribution:
        if key!= -3:
            if category == "preceding":
                print("{:15s} {}".format(preceding_map[key], value))
            elif category == "turn":
                print("{:15s} {}".format(turn_map[key], value))
            elif category == "lane":
                print("{:15s} {}".format(lanemap[key],value))
            else:
                print("{:15s} {}".format(vehiclestate_map[key], value))

    # print class weight
    res_dict,class_weight = calculate_class_weight(label_noonehot)
    print("class_weight:\n",class_weight)

    # load model weight
    if model_select == "FCN":
        model = build_fcn_model(features,class_number= categorymap[category],weights = np.array(class_weight).reshape((1,1,-1)))
    elif model_select == "RESNET":
        model = build_resnet_model(features,class_number= categorymap[category],weights = np.array(class_weight).reshape((1,1,-1)))
    elif model_select == "LSTM":
        model = build_lstm_model(features,class_number= categorymap[category],weights = np.array(class_weight).reshape((1,1,-1)))
    else: # "BLSTM"
        model = build_BLSTM_model(features,class_number= categorymap[category],weights = np.array(class_weight).reshape((1,1,-1)))


    model.load_weights(filename)

    _, model_name, _ = get_filePath_fileName_fileExt(filename)

    # evaluate model
    # loss,accuracy = evaluate_model(filename,features,lateral_label_onehot)

    ## mask the ground truth and predicted labels
    sample_index = 0
    val_feature = features[sample_index:, :, :]
    val_lateral_label = label_noonehot[sample_index:, :]
    val_mask = masks[sample_index:, :].astype(np.bool)
    val_pred = model.predict(val_feature)
    val_pred_cls = np.argmax(val_pred[:,:,:], axis=-1).reshape(sample_number,-1)
    val_true_cls = val_lateral_label

    # mask all predicted labels
    val_pred_cls1 = val_pred_cls[val_mask]
    val_true_cls1 = val_true_cls[val_mask]
    correct_bool = val_pred_cls1 == val_true_cls1

    ## visualize the confusion matrix for all classes
    cf_label = list(np.unique(val_pred_cls1)) + list(np.unique(val_true_cls1))
    cf_labels_num = sorted(np.unique(cf_label))
    cf_label_name = []
    m_map = []
    if category == "preceding":
        for key, value in enumerate(cf_labels_num):
            cf_label_name.append(preceding_map[value])
        m_map = preceding_map
    elif category == "turn":
        for key, value in enumerate(cf_labels_num):
            cf_label_name.append(turn_map[value])
        m_map = turn_map
    elif category == "lane":
        for key, value in enumerate(cf_labels_num):
            cf_label_name.append(lanemap[value])
        m_map = lanemap
    elif category == "vehiclestate":
        for key, value in enumerate(cf_labels_num):
            cf_label_name.append(vehiclestate_map[value])
        m_map = vehiclestate_map

    # plotting sample unit figure, and calculate accuracy, precision,recall,f1score,cm for each sample
    plot_sample(sample_number, val_pred_cls, val_true_cls, masks,model_name,savepath, maneuver_map = m_map)

    cm = confusion_matrix(val_true_cls1, val_pred_cls1)
    print("confusion_matrix,without normalization\n", cm)
    plot_confusion_matrix(cm, cf_label_name,model_name,savepath)

    # # print the average classification_report for each class and save as csv
    print(classification_report(val_true_cls1, val_pred_cls1, digits=4,target_names=cf_label_name))
    report = classification_report(val_true_cls1, val_pred_cls1, digits=4, target_names=cf_label_name, output_dict=True)
    classifaction_report_csv(report,model_name,savepath)

