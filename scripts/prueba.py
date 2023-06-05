## Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import json
import numpy as np

## Parquet Dataset
DATAPATH = '/mnt/f/datasets/isrl_kaggle/'
train_set = pd.read_csv(DATAPATH + 'train.csv')
with open(DATAPATH + 'sign_to_prediction_index_map.json') as f:
    index_map=json.load(f)

ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

LANDMARKS = [61, 40, 37, 0, 267, 270, 291, 91, 84, 17, 314, 321, 78, 81, 13,
             311, 308, 178, 14, 402] + np.arange(468, 512).tolist() + \
             np.arange(522, 543).tolist()

data_size = len(train_set)
X = np.empty((data_size, 85, 85, 3))
y = np.empty((data_size, 250), dtype=int)

for index, row in train_set.iterrows():
    data = load_relevant_data_subset(DATAPATH + row.path)
    data = np.nan_to_num(data[:, LANDMARKS, :], 0)
    data = tf.image.resize(data, size=(85, 85), method='nearest')
    X[index, ] = data.numpy()
    y[index, ] = tf.keras.utils.to_categorical(index_map[row.sign], num_classes=250)
    # break

print(len(X), X[0].shape, len(y), y[0].shape)


np.save(DATAPATH + 'X_data.pickle', X)
np.save(DATAPATH + 'y_data.pickle', y)
