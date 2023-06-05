import tensorflow as tf
import pandas as pd
import json
import numpy as np
import pickle


# Parquet Dataset
DATAPATH = '/mnt/f/datasets/isrl_kaggle/'
train_set = pd.read_csv(DATAPATH + 'train.csv')
with open(DATAPATH + 'sign_to_prediction_index_map.json') as f:
    index_map = json.load(f)

ROWS_PER_FRAME = 543  # number of landmarks per frame


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


X = []
y = []
for index, row in train_set.iterrows():
    X.append(load_relevant_data_subset(DATAPATH + row.path))
    y.append(tf.keras.utils.to_categorical(index_map[row.sign],
                                           num_classes=250))
    if index % 100 == 0:
        print(f'{index} loaded')
    if (index+1) % 10000 == 0:
        with open(f'X_data_{index}.pickle', 'wb') as f:
            pickle.dump(X, f)
        print(f'X_{index} saved')

        with open(f'y_data_{index}.pickle', 'wb') as f:
            pickle.dump(y, f)
        print(f'y_{index} saved')
        X = []
        y = []

print(len(X), X[0].shape, len(y), y[0].shape)


with open('X_data.pickle', 'wb') as f:
    pickle.dump(X, f)
print('X saved')

with open('y_data.pickle', 'wb') as f:
    pickle.dump(y, f)
print('y saved')
