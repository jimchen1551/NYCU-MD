import os
import numpy as np
import scipy.io
from tensorflow.keras.utils import to_categorical

sig, tag_0, tag_1, tag_2 = [], [], [], []
for i in os.listdir('./YM'): 
    sig_0, act_0, act_1, act_2 = [], [], [], []
    if i != '.DS_Store': 
        for j in os.listdir('./YM/'+i): 
            if j != '.DS_Store': 
                print(i + '/' + j)
                mat = scipy.io.loadmat('./YM/'+i+"/"+j)
                print(mat['A_X'].shape)
