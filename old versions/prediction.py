import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'
import numpy as np
import TypeMapping
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from ImportData import scenarios_data, sample_num, top_k_scenario
from collections import Counter
from main2 import build_model


def evaluate_model(model,features,lateral_label):
    x_test, y_test = features[4:,:,:],lateral_label[4:,:,:]
    optm = Adam(lr=1e-3)

    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_model("model_last.h5")
    loss,accuracy = model.evaluate(x_test,y_test)

    return accuracy


if __name__ == "__main__":
    ##----------------- scenario_data from importData.py
    sample_number = sample_num
    max_timestep = 3500
    features = np.zeros((sample_number, max_timestep, 10), dtype=np.float32)
    labels   = np.zeros((sample_number, max_timestep, 2), dtype=np.int64)
    masks    = np.zeros((sample_number, max_timestep), dtype=np.int64)
    index    = 0
    for i, scenario_data in enumerate(scenarios_data):
        if i == top_k_scenario: break
        for key in scenario_data:
            signal_value = scenario_data[key]['signals'].values[:, list(range(0,7))+list(range(10,13))]
            maneuver_temp = scenario_data[key]['maneuverIdentification'].values[:,1:3]
            for idx, item in enumerate(maneuver_temp):
                lateral = TypeMapping.lateral_id_mapping[item[0]]
                longtudinal = TypeMapping.longitudinal_id_mapping[item[1]]
                labels[index, idx, :] = [lateral, longtudinal]
            features[index, :len(signal_value), :] = signal_value # shape, (num_sample, max_timestep, 10)
            masks[index, :len(signal_value)] = 1 # shape, (num_sample, max_timestep)
            index += 1

    lateral_label_noonehot = labels[:, :, 0]

    test_sample_distribution  = Counter(lateral_label_noonehot.reshape(-1).tolist())

    print("\nlabel appear in test:\n")
    for key, value in test_sample_distribution.items():
        if key:
            print("{:15s} {}".format(TypeMapping.lateral_distribution[key], value))

    # load full saved model
    # model = load_model("weights/weights.h5")

    # load model weight
    model = build_model(features)
    model.load_weights("weights/weights_best.h5")

    # acc = evaluate_model(model,features,lateral_label)
    sample_index = 0
    val_feature = features[sample_index:, :, :]
    val_lateral_label = lateral_label_noonehot[sample_index:, :]
    val_mask = masks[sample_index:, :].astype(np.bool)

    val_pred = model.predict(val_feature)
    val_pred_cls = np.argmax(val_pred[:,:,1:], axis=-1)
    val_true_cls = val_lateral_label

    # plotting sample unit figure
    for index in range(sample_number):
        val_pred_cls_sample = val_pred_cls[index]
        val_true_cls_sample = val_true_cls[index]
        mask_sample = masks[index].astype(np.bool)
        val_pred_cls_sample = val_pred_cls_sample[mask_sample]
        val_true_cls_sample = val_true_cls_sample[mask_sample]
        correct_bool = val_pred_cls_sample == val_true_cls_sample
        accuracy = correct_bool.astype(np.int).sum() / len(correct_bool)

        plt.plot(val_pred_cls_sample, 'g-.', label="predict")
        plt.plot(val_true_cls_sample, 'r--', label="gt")
        plt.legend()
        plt.xlabel("timesteps")
        plt.ylabel("maneuver")
        plt.title("sample accuracy {:.4f}".format(accuracy))
        plt.savefig("plots/{}_sample_{}".format("scenario188", index))
        plt.waitforbuttonpress()
        plt.close()

    val_pred_cls = val_pred_cls[val_mask]
    val_true_cls = val_true_cls[val_mask]
    correct_bool = val_pred_cls == val_true_cls
    accuracy = correct_bool.astype(np.int).sum() / len(correct_bool)

    print("test pred_cls index\n {}".format(val_pred_cls))
    print("test true_cls index\n {}".format(val_true_cls))
    print("test accuracy: {}".format(accuracy))

    # output prediction result in txt file
    out_path = "gt_predictions.txt"
    file = open(out_path, "w")
    file.write("test pred_cls index\n")
    file.write(", ".join(map(str, val_pred_cls)))
    file.write("\ntest true_cls index\n")
    file.write(", ".join(map(str, val_true_cls)))
    file.write("\ncorrect_bool\n")
    file.write(", ".join(map(str, correct_bool)))
    file.write("\ntest accuracy: {}".format(accuracy))
    file.close()