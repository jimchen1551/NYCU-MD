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

import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpu) == 1
tf.config.experimental.set_memory_growth(gpu[0], True)

np.random.seed(10)

function = []
with open('function.txt') as f:
    rfunction = f.readlines()
    for i in rfunction:
        x = i.replace('\n', '')
        function.append(x.split(','))

parameter = []
with open('parameter.txt') as f:
    rpara = f.readlines()
    for i in rpara:
        x = i.replace('\n', '')
        parameter.append(x.split(' '))
for i in range(3):
    for j in range(3):
        parameter[i][j] = parameter[i][j].split(',')
parameter[3][0] = parameter[3][0].split(',')

signal, label_0, label_1, label_2 = [], [], [], []
with open('data.npy', 'rb') as f:
    for i in range(3):
        signal.append(np.load(f))
        label_0.append(np.load(f))
        label_1.append(np.load(f))
        label_2.append(np.load(f))

for i in function:
    for j in parameter[:3]:
        for k in parameter[3][0][0]:
            for l in range(3):
                model_0_0 = Sequential()
                model_0_0.add(ZeroPadding2D(padding=(2, 2)))
                model_0_0.add(Conv2D(int(j[0][0]), kernel_size=(5, 1), padding='valid', activation=i[0], input_shape=(101, 3, 1)))
                model_0_0.add(ZeroPadding2D(padding=(2, 2)))
                model_0_0.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
                model_0_0.add(ZeroPadding2D(padding=(2, 2)))
                model_0_0.add(Conv2D(int(j[0][1]), kernel_size=(5, 1), padding='valid', activation=i[0]))
                model_0_0.add(ZeroPadding2D(padding=(2, 2)))
                model_0_0.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
                model_0_0.add(ZeroPadding2D(padding=(2, 2)))
                model_0_0.add(Conv2D(int(j[0][2]), kernel_size=(5, 1), padding='valid', activation=i[0]))
                model_0_0.add(Dropout(int(k)))
                model_0_0.add(Flatten())
                model_0_0.add(Dense(10, kernel_regularizer="l2", activation=i[1]))
                model_0_0.compile(loss="categorical_crossentropy", optimizer=i[2], metrics=["accuracy"])
                history_0_0 = model_0_0.fit(signal[1], label_2[1], validation_data=(signal[2], label_2[2]), epochs=300, batch_size=64, verbose=False)

                with open('model_2.txt', 'a') as f:
                    f.write('model_2_0\t'+function.index(i)+'\t'+parameter.index(j)+'\t'+k+'\t'+str(l)+'\t')
                    loss, accuracy = model_0_0.evaluate(signal[1], label_2[1], verbose=False)
                    f.write("Training accuracy = {:.2f}".format(accuracy)+'\t')
                    loss, accuracy = model_0_0.evaluate(signal[0], label_2[0], verbose=False)
                    f.write("Testing accuracy = {:.2f}".format(accuracy)+'\n')

                loss = history_0_0.history["loss"]
                epochs = range(1, len(loss)+1)
                val_loss = history_0_0.history["val_loss"]
                plt.plot(epochs, loss, "bo-", label="Training Loss")
                plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
                plt.title('model_2_0\t'+function.index(i)+'\t'+parameter.index(j)+'\t'+k+'\t'+str(l))
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig('model_2_0/'+function.index(i)+'_'+parameter.index(j)+'_'+k+'_'+str(l)+'_loss')
                plt.clf()
                acc = history_0_0.history["accuracy"]
                epochs = range(1, len(acc)+1)
                val_acc = history_0_0.history["val_accuracy"]
                plt.plot(epochs, acc, "bo-", label="Training Acc")
                plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
                plt.title('model_2_0\t'+function.index(i)+'\t'+parameter.index(j)+'\t'+k+'\t'+str(l))
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.savefig('model_2_0/'+function.index(i)+'_'+parameter.index(j)+'_'+k+'_'+str(l)+'_accuracy')
                plt.clf()

for i in function:
    for j in parameter[:3]:
        for k in parameter[3][0][0]:
            for l in range(3):
                model_0_1 = Sequential()
                model_0_1.add(ZeroPadding2D(padding=(2, 2)))
                model_0_1.add(Conv2D(int(j[1][0]), kernel_size=(5, 1), padding='valid', activation=i[0], input_shape=(101, 3, 1)))
                model_0_1.add(ZeroPadding2D(padding=(2, 2)))
                model_0_1.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
                model_0_1.add(ZeroPadding2D(padding=(2, 2)))
                model_0_1.add(Conv2D(int(j[1][1]), kernel_size=(5, 1), padding='valid', activation=i[0]))
                model_0_1.add(ZeroPadding2D(padding=(2, 2)))
                model_0_1.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
                model_0_1.add(ZeroPadding2D(padding=(2, 2)))
                model_0_1.add(Conv2D(int(j[1][2]), kernel_size=(5, 1), padding='valid', activation=i[0]))
                model_0_1.add(ZeroPadding2D(padding=(2, 2)))
                model_0_1.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
                model_0_1.add(ZeroPadding2D(padding=(2, 2)))
                model_0_1.add(Conv2D(int(j[1][3]), kernel_size=(5, 1), padding='valid', activation=i[0]))
                model_0_1.add(Dropout(int(k)))
                model_0_1.add(Flatten())
                model_0_1.add(Dense(10, kernel_regularizer="l2", activation=i[1]))
                model_0_1.compile(loss="categorical_crossentropy", optimizer=i[2], metrics=["accuracy"])
                history_0_1 = model_0_1.fit(signal[1], label_2[1], validation_data=(signal[2], label_2[2]), epochs=300, batch_size=64, verbose=False)

                with open('model_2.txt', 'a') as f:
                    f.write('model_2_1\t'+function.index(i)+'\t'+parameter.index(j)+'\t'+k+'\t'+str(l)+'\t')
                    loss, accuracy = model_0_1.evaluate(signal[1], label_2[1], verbose=False)
                    f.write("Training accuracy = {:.2f}".format(accuracy)+'\t')
                    loss, accuracy = model_0_1.evaluate(signal[0], label_2[0], verbose=False)
                    f.write("Testing accuracy = {:.2f}".format(accuracy)+'\n')

                loss = history_0_1.history["loss"]
                epochs = range(1, len(loss)+1)
                val_loss = history_0_1.history["val_loss"]
                plt.plot(epochs, loss, "bo-", label="Training Loss")
                plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
                plt.title('model_2_1\t'+function.index(i)+'\t'+parameter.index(j)+'\t'+k+'\t'+str(l))
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig('model_2_1/'+function.index(i)+'_'+parameter.index(j)+'_'+k+'_'+str(l)+'_loss')
                plt.clf()
                acc = history_0_1.history["accuracy"]
                epochs = range(1, len(acc)+1)
                val_acc = history_0_1.history["val_accuracy"]
                plt.plot(epochs, acc, "bo-", label="Training Acc")
                plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
                plt.title('model_2_1\t'+function.index(i)+'\t'+parameter.index(j)+'\t'+k+'\t'+str(l))
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.savefig('model_2_1/'+function.index(i)+'_'+parameter.index(j)+'_'+k+'_'+str(l)+'_accuracy')
                plt.clf()

for i in function:
    for j in parameter[:3]:
        for k in parameter[3][0][0]:
            for l in range(3):
                model_0_2 = Sequential()
                model_0_2.add(ZeroPadding2D(padding=(2, 2)))
                model_0_2.add(Conv2D(int(j[2][0]), kernel_size=(5, 1), padding='valid', activation=i[0], input_shape=(101, 3, 1)))
                model_0_2.add(ZeroPadding2D(padding=(2, 2)))
                model_0_2.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
                model_0_2.add(ZeroPadding2D(padding=(2, 2)))
                model_0_2.add(Conv2D(int(j[2][1]), kernel_size=(5, 1), padding='valid', activation=i[0]))
                model_0_2.add(ZeroPadding2D(padding=(2, 2)))
                model_0_2.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
                model_0_2.add(ZeroPadding2D(padding=(2, 2)))
                model_0_2.add(Conv2D(int(j[2][2]), kernel_size=(5, 1), padding='valid', activation=i[0]))
                model_0_2.add(ZeroPadding2D(padding=(2, 2)))
                model_0_2.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
                model_0_2.add(ZeroPadding2D(padding=(2, 2)))
                model_0_2.add(Conv2D(int(j[2][3]), kernel_size=(5, 1), padding='valid', activation=i[0]))
                model_0_2.add(ZeroPadding2D(padding=(2, 2)))
                model_0_2.add(MaxPooling2D(pool_size=(5, 1), padding='valid'))
                model_0_2.add(ZeroPadding2D(padding=(2, 2)))
                model_0_2.add(Conv2D(int(j[2][4]), kernel_size=(5, 1), padding='valid', activation=i[0]))
                model_0_2.add(Dropout(int(k)))
                model_0_2.add(Flatten())
                model_0_2.add(Dense(10, kernel_regularizer="l2", activation=i[1]))
                model_0_2.compile(loss="categorical_crossentropy", optimizer=i[2], metrics=["accuracy"])
                history_0_2 = model_0_2.fit(signal[1], label_2[1], validation_data=(signal[2], label_2[2]), epochs=300, batch_size=64, verbose=False)


                with open('model_2.txt', 'a') as f:
                    f.write('model_2_2\t'+function.index(i)+'\t'+parameter.index(j)+'\t'+k+'\t'+str(l)+'\t')
                    loss, accuracy = model_0_2.evaluate(signal[1], label_2[1], verbose=False)
                    f.write("Training accuracy = {:.2f}".format(accuracy)+'\t')
                    loss, accuracy = model_0_2.evaluate(signal[0], label_2[0], verbose=False)
                    f.write("Testing accuracy = {:.2f}".format(accuracy)+'\n')

                loss = history_0_2.history["loss"]
                epochs = range(1, len(loss)+1)
                val_loss = history_0_2.history["val_loss"]
                plt.plot(epochs, loss, "bo-", label="Training Loss")
                plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
                plt.title('model_2_2\t'+function.index(i)+'\t'+parameter.index(j)+'\t'+k+'\t'+str(l))
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig('model_2_2/'+function.index(i)+'_'+parameter.index(j)+'_'+k+'_'+str(l)+'_loss')
                plt.clf()
                acc = history_0_2.history["accuracy"]
                epochs = range(1, len(acc)+1)
                val_acc = history_0_2.history["val_accuracy"]
                plt.plot(epochs, acc, "bo-", label="Training Acc")
                plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
                plt.title('model_2_2\t'+function.index(i)+'\t'+parameter.index(j)+'\t'+k+'\t'+str(l))
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.savefig('model_2_2/'+function.index(i)+'_'+parameter.index(j)+'_'+k+'_'+str(l)+'_accuracy')
                plt.clf()