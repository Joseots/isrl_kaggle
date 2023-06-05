import tensorflow as tf
import pandas as pd
import numpy as np
from .config import ROWS_PER_FRAME, LANDMARKS, DATAPATH


class LoadData():
    def __init__(self) -> None:
        pass

    def load_relevant_data_subset(self, pq_path):
        data_columns = ['x', 'y', 'z']
        data = pd.read_parquet(pq_path, columns=data_columns)
        n_frames = int(len(data) / ROWS_PER_FRAME)
        data = data.values.reshape(n_frames, ROWS_PER_FRAME,
                                   len(data_columns))
        return data.astype(np.float32)


# Data Generator
class DataGenerator(tf.keras.utils.Sequence):
    '''Generates data for Keras'''
    def __init__(self, list_IDs, train_set, labels, batch_size=32,
                 dim=(160, 85), n_channels=3, n_classes=250, shuffle=True):
        '''Initialization'''
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.train_set = train_set
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        '''Generates data containing batch_size samples'''
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            current_path = DATAPATH + self.train_set.path[ID]
            data = self.load_relevant_data_subset(current_path)
            data = np.nan_to_num(data[:, LANDMARKS, :], 0)
            data = tf.image.resize(data, size=self.dim, method='nearest')
            X[i, ] = data.numpy()

            # Store class
            y[i] = self.labels[self.train_set.sign[ID]]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
