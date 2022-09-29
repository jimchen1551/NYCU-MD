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
                mat = scipy.io.loadmat('./YM/'+i+"/"+j)
                sig_0.append(np.lib.pad(np.concatenate((mat['A_X'], mat['A_Y'], mat['A_Z']), axis=1), ((0, 101 - mat['A_X'].size), (0, 0)), 'constant', constant_values=0).reshape(1, 101, 3, 1))
                if mat['Active']<=9: 
                    k = 0
                else: 
                    k = 1
                act_0.append(k)
                if mat['Active']>=10:
                    k = 0
                else: 
                    k = mat['Active']
                act_1.append(k)
                if mat['Active']<=9:
                    k = 0
                elif mat['Active']==10:
                    k = 1
                elif mat['Active']==11:
                    k = 2
                elif mat['Active']==12:
                    k = 3
                elif mat['Active']==13:
                    k = 4
                elif mat['Active']==14:
                    k = 5
                elif mat['Active']==15:
                    k = 6
                act_2.append(k)
        sig.append(np.asarray(np.concatenate(tuple(sig_0), axis=0)).astype(np.float16))
        tag_0.append(np.array(act_0).reshape(len(act_0), 1))
        tag_1.append(np.array(act_1).reshape(len(act_1), 1))
        tag_2.append(np.array(act_2).reshape(len(act_2), 1))

for i in range(0, len(tag_0)):
    tag_0.insert(i, to_categorical(tag_0[i]))
    tag_0.pop(i+1)
    tag_1.insert(i, to_categorical(tag_1[i]))
    tag_1.pop(i+1)
    tag_2.insert(i, to_categorical(tag_2[i]))
    tag_2.pop(i+1)

with open('data.npy', 'wb') as f:
    for i in range(3):
        np.save(f, sig[i])
        np.save(f, tag_0[i])
        np.save(f, tag_1[i])
        np.save(f, tag_2[i])