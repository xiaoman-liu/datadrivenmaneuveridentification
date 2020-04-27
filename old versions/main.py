from ImportData import data
import numpy as np
import pandas as pd
import TypeMapping
import keras as K
from  keras.preprocessing import sequence
from keras.layers import Masking
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

print(data)
length = []
sequences = list()
X_train = []
Y_train = []
sample_number = 6
max_timestep = 1000
# TODO: car type in feature?
# TODO: check one hot.
features = np.zeros((sample_number, max_timestep, 10), dtype=np.float32)
labels   = np.zeros((sample_number, max_timestep, 2), dtype=np.int64)
masks    = np.zeros((sample_number, max_timestep), dtype=np.int64)
for index, key in enumerate(data):
    # signal_value = data[key]['signals'].values[:,[0,1,2,3,4,5,6,10,11,12]]
    # min_max = preprocessing.MinMaxScaler()
    # sample_min_max = min_max.fit_transform(signal_value)
    # sample = np.hstack((sample_min_max,maneuver))
    # sequences.append(sample)

    signal_value = data[key]['signals'].values[:, list(range(0,7))+list(range(10,13))]
    maneuver_temp = data[key]['maneuverIdentification'].values[:,1:3]
    for idx, item in enumerate(maneuver_temp):
        lateral = TypeMapping.lateral_id_mapping[item[0]]
        longtudinal = TypeMapping.longitudinal_id_mapping[item[1]]
        labels[index, idx, :] = [lateral, longtudinal]
    features[index, :len(signal_value), :] = signal_value # shape, (num_sample, max_timestep, 10)
    masks[index, :len(signal_value)] = 1 # shape, (num_sample, max_timestep)

lateral_label = labels[:, :, 0]
lateral_label = K.utils.to_categorical(lateral_label)



# en = preprocessing.OneHotEncoder(sparse = False)
# EN1en = preprocessing.OneHotEncoder(sparse = False)
# feature_x = en.fit_transform()
# label_y = EN1en.fit_transform()


len_sequence = []
for i in sequences:
    len_sequence.append(len(i))
pd.Series(len_sequence).describe()

##pad into the same length

to_pad = max(len_sequence)
new_sep = []
for one_seq in sequences:
    len_one = len(one_seq)


    n = to_pad - len_one
    if n:
        to_concat = -100 * np.ones((n,12),dtype = float).reshape(-1,n).transpose()
        new_one_seq =np.concatenate([one_seq,to_concat])
        new_sep.append(new_one_seq)
    else:
        new_sep.append(one_seq)

final_seq = np.stack(new_sep)#add a new dimension

final_data = sequence.pad_sequences(final_seq,maxlen=to_pad,dtype = float,padding = 'post',truncating = 'post')
num_features = len(final_data) - 3

#build modelcompile
def build_model():

    model = Sequential()
    model.add(Masking(mask_value= -100,input_shape=(to_pad,num_features)))
    model.add(LSTM(128,return_sequences = True))
    model.add(LSTM(64,return_sequences = True))
    model.add(LSTM(1,return_sequences = True))
    model.add(Dense(8,activation="softmax"))
    optm = Adam(lr=0.0001)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = build_model()
model.summary()
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
                                  factor=0.7937, cooldown=0, min_lr=1e-4, verbose=2)
epochs = 10
callback_list = [reduce_lr]
hist = model.fit(X_train, Y_train, batch_size=1, epochs=epochs, callbacks=callback_list,
                  verbose=2, validation_split = 0.2)



len_sequence = []
for i in sequences:
    len_sequence.append(len(i))
pd.Series(len_sequence).describe()

##pad into the same length










to_pad = 540
new_sep = []
for one_seq in sequences:
    len_one = len(one_seq)
    last = one_seq[-1]
    n = to_pad - len_one
    if n:
        to_concat = np.repeat(last,n).reshape(-1,n).transpose()
        new_one_seq =np.concatenate([one_seq,to_concat])
        new_sep.append(new_one_seq)
    else:
        new_sep.append(one_seq)

final_seq = np.stack(new_sep)#add a new dimension
seq_len = 540
final_seq = sequence.pad_sequences(final_seq,maxlen=seq_len,dtype = float,padding = 'post',truncating = 'post')








# n_sample = 0
# for key in data:
#     n_sample+=len(data[key])
#
# for key in data:
#     scenarios_data = data[key]
#     for key in scenarios_data:
#         a = scenarios_data[key]['signals']




