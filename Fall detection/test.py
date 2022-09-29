import numpy as np
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt

np.random.seed(10)

signal, label_0, label_1, label_2 = [], [], [], []
with open('data.npy', 'rb') as f:
    for i in range(3):
        signal.append(np.load(f))
        label_0.append(np.load(f))
        label_1.append(np.load(f))
        label_2.append(np.load(f))

print(label_2[1].shape)