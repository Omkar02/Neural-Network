from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from sklearn.model_selection import train_test_split
from tflearn.layers.estimator import regression
from vectors import lable
import pickle
import numpy as np
import os
from PIL import Image


#******************* Data loading and preprocessing*********************************************************************
with open('CHAR_IMG_flat.pkl', 'rb') as f:
    X= pickle.load(f)
print(X)
Y = lable
print(Y)
X, Y = shuffle(X, Y)
X, testX, Y, testY = train_test_split(X, Y, test_size=0.3, random_state=0)
X = np.reshape(X,[-1, 28, 28, 1])
testX, testY = shuffle(testX, testY)
testX = np.reshape(testX,[-1, 28, 28, 1])
#******************* Network *******************************************************************************************
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.2)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.2)
network = fully_connected(network, 36, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.1,
                     loss='categorical_crossentropy', name='target')

#******************* Training ******************************************************************************************
model = tflearn.DNN(network, tensorboard_verbose=3)
model.fit({'input': X}, {'target': Y}, n_epoch=60,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=1000, show_metric=True, run_id='Some_detection_needed!')

model.save('model.tflearn')
#model.load('model.tflearn')
#******************* predict *******************************************************************************************
# data_path = 'test'
# for dataset in os.listdir(data_path):
#     img = Image.open(os.path.join(data_path, dataset)).convert('L')
#     img = img.resize((28, 28), Image.ANTIALIAS)
#     img = np.reshape(img, (-1,28, 28, 1))
#     pre = model.predict(img)
#     print('Value = ',pre)
# 
#     print('**************')
