from keras import backend as K
from keras.layers import Layer
import keras.layers as kl
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.python.keras.layers import Flatten

input_shape = (28, 28, 1)

conv = models.Sequential()

# 2/ 2 filters
conv.add(layers.Conv2D(2, (5, 5), activation='relu', input_shape=input_shape))

# 3/ pooling
conv.add(layers.MaxPooling2D((2, 2)))

# 4/ 4 filters
conv.add(layers.Conv2D(4, (3, 3, 2), activation='sigmoid'))

# 5/ pooling
conv.add(layers.MaxPooling2D((2, 2)))

# 6/flatten
conv = (Flatten()(conv))
