import os
import pandas as pd
import numpy as np


path = "./data/DR_USA_Intersection_EP0"
files = sorted(os.listdir(path))
data_list = []
max_length = 0
min_length = 800
length = []
for file in files:
    if file.endswith(".csv"):
        file_path = os.path.join(path,file)
        data = pd.read_csv(file_path)
        data = data.fillna(0).values

        sample = []
        for line in data:

            if not sample:
                sample.append(line)
            else:
                if line[0] == sample[-1][0]:
                    sample.append(line)
                else:
                    data_list.append(np.array(sample))
                    max_length = max(max_length, len(sample))
                    min_length = min(min_length, len(sample))
                    length.append(len(sample))
                    sample = []
# data_list = np.array(data_list)
sample_num = len(data_list)
