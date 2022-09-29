import os
import numpy as np
import scipy.io
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

np.random.seed(10)

sig, tag_0, tag_1, tag_2 = [], [], [], []
for i in os.listdir('./UP'): 
    sig_0, act_0, act_1, act_2 = [], [], [], []
    if i != '.DS_Store': 
        for j in os.listdir('./UP/'+i): 
            if i != '.DS_Store': 
                mat = scipy.io.loadmat('./UP/'+i+"/"+j)
                sig_0.append(np.lib.pad(np.concatenate((mat['trainingGx'], mat['trainingGy'], mat['trainingGz']), axis=1), ((0, 126 - mat['trainingGx'].size), (0, 0)), 'constant', constant_values=0).reshape(1, 126, 3, 1))
                if mat['Activity']>=6: 
                    k = 0
                else: 
                    k = 1
                act_0.append(k)
                if mat['Activity']<=5:
                    k = 0
                elif mat['Activity']==6:
                    k = 1
                elif mat['Activity']==7:
                    k = 2
                elif mat['Activity']==8:
                    k = 3
                elif mat['Activity']==9:
                    k = 4
                elif mat['Activity']==10:
                    k = 5
                elif mat['Activity']==11:
                    k = 6
                act_1.append(k)
                if mat['Activity']>=6:
                    k = 0
                else: 
                    k = mat['Activity']
                act_2.append(k)
        sig.append(np.asarray(np.concatenate(tuple(sig_0), axis=0)).astype(np.float16))
        tag_0.append(np.array(act_0).reshape(len(act_0), 1))
        tag_1.append(np.array(act_1).reshape(len(act_1), 1))
        tag_2.append(np.array(act_2).reshape(len(act_2), 1))

#One-hot encoding
for i in range(0, len(tag_0)):
    tag_0.insert(i, to_categorical(tag_0[i]))
    tag_0.pop(i+1)
    tag_1.insert(i, to_categorical(tag_1[i]))
    tag_1.pop(i+1)
    tag_2.insert(i, to_categorical(tag_2[i]))
    tag_2.pop(i+1)

signal = []
label_0, label_1, label_2 = [], [], []
signal.append(sig[0])
signal.append(sig[1])
signal.append(sig[2])
label_0.append(tag_0[0])
label_0.append(tag_0[1])
label_0.append(tag_0[2])
label_1.append(tag_1[0])
label_1.append(tag_1[1])
label_1.append(tag_1[2])
label_2.append(tag_2[0])
label_2.append(tag_2[1])
label_2.append(tag_2[2])

model_0 = Sequential()
model_0.add(ZeroPadding2D(padding=(2, 2)))
model_0.add(Conv2D(16, kernel_size=(5, 1), padding='valid', activation='elu', input_shape=(101, 3, 1)))
model_0.add(ZeroPadding2D(padding=(2, 2)))
model_0.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
model_0.add(ZeroPadding2D(padding=(2, 2)))
model_0.add(Conv2D(32, kernel_size=(5, 1), padding='valid', activation='elu'))
model_0.add(ZeroPadding2D(padding=(2, 2)))
model_0.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
model_0.add(ZeroPadding2D(padding=(2, 2)))
model_0.add(Conv2D(64, kernel_size=(5, 1), padding='valid', activation='elu'))
model_0.add(ZeroPadding2D(padding=(2, 2)))
model_0.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
model_0.add(ZeroPadding2D(padding=(2, 2)))
model_0.add(Conv2D(128, kernel_size=(5, 1), padding='valid', activation='elu'))
model_0.add(Dropout(0.5))
model_0.add(Flatten())
model_0.add(Dense(2, kernel_regularizer="l2", activation="softmax"))
model_0.compile(loss="categorical_crossentropy", optimizer='adagrad', metrics=["accuracy"])
history_0 = model_0.fit(signal[1], label_0[1], validation_data=(signal[2], label_0[2]), epochs=1000, batch_size=64, verbose=False)
print("\nTesting ...")
loss, accuracy = model_0.evaluate(signal[1], label_0[1], verbose=False)
print("Taining accuracy = {:.2f}".format(accuracy))
loss, accuracy = model_0.evaluate(signal[0], label_0[0], verbose=False)
print("Testing accuracy = {:.2f}".format(accuracy))
# converter = tf.lite.TFLiteConverter.from_keras_model(model_0)
# tflite_model = converter.convert()
# with tf.io.gfile.GFile('model_0.tflite', 'wb') as f:
#     f.write(tflite_model)
loss = history_0.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history_0.history["val_loss"]
plt.plot(epochs, loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
acc = history_0.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history_0.history["val_accuracy"]
plt.plot(epochs, acc, "bo-", label="Training Acc")
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

model_1 = Sequential()
model_1.add(ZeroPadding2D(padding=(2, 2)))
model_1.add(Conv2D(16, kernel_size=(5, 1), padding='valid', activation='relu', input_shape=(101, 3, 1)))
model_1.add(ZeroPadding2D(padding=(2, 2)))
model_1.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
model_1.add(ZeroPadding2D(padding=(2, 2)))
model_1.add(Conv2D(32, kernel_size=(5, 1), padding='valid', activation='relu'))
model_1.add(ZeroPadding2D(padding=(2, 2)))
model_1.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
model_1.add(ZeroPadding2D(padding=(2, 2)))
model_1.add(Conv2D(64, kernel_size=(5, 1), padding='valid', activation='relu'))
model_1.add(ZeroPadding2D(padding=(2, 2)))
model_1.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
model_1.add(ZeroPadding2D(padding=(2, 2)))
model_1.add(Conv2D(128, kernel_size=(5, 1), padding='valid', activation='relu'))
model_1.add(Dropout(0.5))
model_1.add(Flatten())
model_1.add(Dense(7, kernel_regularizer="l2", activation="sigmoid"))
model_1.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=["accuracy"])
history_1 = model_1.fit(signal[1], label_1[1], validation_data=(signal[2], label_1[2]), epochs=1000, batch_size=64, verbose=False)
print("\nTesting ...")
loss, accuracy = model_1.evaluate(signal[1], label_1[1], verbose=False)
print("Taining accuracy = {:.2f}".format(accuracy))
loss, accuracy = model_1.evaluate(signal[0], label_1[0], verbose=False)
print("Testing accuracy = {:.2f}".format(accuracy))
# converter = tf.lite.TFLiteConverter.from_keras_model(model_1)
# tflite_model = converter.convert()
# with tf.io.gfile.GFile('model_1.tflite', 'wb') as f:
#     f.write(tflite_model)
loss = history_1.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history_1.history["val_loss"]
plt.plot(epochs, loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
acc = history_1.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history_1.history["val_accuracy"]
plt.plot(epochs, acc, "bo-", label="Training Acc")
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

model_2 = Sequential()
model_2.add(ZeroPadding2D(padding=(2, 2)))
model_2.add(Conv2D(16, kernel_size=(5, 1), padding='valid', activation='relu', input_shape=(101, 3, 1)))
model_2.add(ZeroPadding2D(padding=(2, 2)))
model_2.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
model_2.add(ZeroPadding2D(padding=(2, 2)))
model_2.add(Conv2D(32, kernel_size=(5, 1), padding='valid', activation='relu'))
model_2.add(ZeroPadding2D(padding=(2, 2)))
model_2.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
model_2.add(ZeroPadding2D(padding=(2, 2)))
model_2.add(Conv2D(64, kernel_size=(5, 1), padding='valid', activation='relu'))
model_2.add(ZeroPadding2D(padding=(2, 2)))
model_2.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
model_2.add(ZeroPadding2D(padding=(2, 2)))
model_2.add(Conv2D(128, kernel_size=(5, 1), padding='valid', activation='relu'))
model_2.add(Dropout(0.5))
model_2.add(Flatten())
model_2.add(Dense(6, kernel_regularizer="l2", activation="softmax"))
model_2.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=["accuracy"])
history_2 = model_2.fit(signal[1], label_2[1], validation_data=(signal[2], label_2[2]), epochs=1000, batch_size=64, verbose=False)
print("\nTesting ...")
loss, accuracy = model_2.evaluate(signal[1], label_2[1], verbose=False)
print("Taining accuracy = {:.2f}".format(accuracy))
loss, accuracy = model_2.evaluate(signal[0], label_2[0], verbose=False)
print("Testing accuracy = {:.2f}".format(accuracy))
# converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
# tflite_model = converter.convert()
# with tf.io.gfile.GFile('model_2.tflite', 'wb') as f:
#     f.write(tflite_model)
loss = history_2.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history_2.history["val_loss"]
plt.plot(epochs, loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
acc = history_2.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history_2.history["val_accuracy"]
plt.plot(epochs, acc, "bo-", label="Training Acc")
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ans_1 = model_1.predict(signal[0], verbose=False)
# fallsig = []
# falllab = []
# fallinfo = []
# for i in range(len(ans_1)):
#     if ans_1[i][6] == max(ans_1[i]):
#         fallsig.append(signal[0][i])
# ans_2 = model_2.predict(fallsig, verbose=False)
# print(ans_2)