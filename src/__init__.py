import numpy as np
import json
import pandas as pd

DATAPATH = '/mnt/f/datasets/isrl_kaggle/'
ROWS_PER_FRAME = 543  # number of landmarks per frame
LANDMARKS = [61, 40, 37, 0, 267, 270, 291, 91, 84, 17, 314, 321, 78, 81, 13,
             311, 308, 178, 14, 402] + np.arange(468, 512).tolist() + \
             np.arange(522, 543).tolist()
train_set = pd.read_csv(DATAPATH + 'train.csv')
with open(DATAPATH + 'sign_to_prediction_index_map.json') as f:
    index_map = json.load(f)
