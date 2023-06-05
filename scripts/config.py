import numpy as np
import json
import pandas as pd


DATAPATH = '../data/'
ROWS_PER_FRAME = 543  # number of landmarks per frame
train_set = pd.read_csv(DATAPATH + 'train.csv')

with open(DATAPATH + 'sign_to_prediction_index_map.json') as f:
    index_map = json.load(f)

LANDMARKS = {
    'face': [61, 40, 37, 0, 267, 270, 291, 91, 84, 17, 314, 321, 78, 81, 13,
             311, 308, 178, 14, 402],
    'left_hand': np.arange(468, 489),
    'pose': np.arange(490, 512),
    'right_hand': np.arange(522, 543)
    }
