# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 16:21:06 2017

@author: 44ry8x1
"""

import pickle
import bokeh

pkl_file = open('data.pkl', 'rb')
pkl_file1 = open('train_data.pkl', 'rb')
track_train = pickle.load(pkl_file)
df_data = pickle.load(pkl_file1)

# outlier_keys: [486, 492, 1824, 2018]

white, black = {}, {}
for k, v in track_train.items():
    v['target_x'] = df_data['target_x'][k]
    v['target_y'] = df_data['target_y'][k]
    if k < 2600:
        white[k] = v
    else:
        black[k] = v

