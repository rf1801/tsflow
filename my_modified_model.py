import numpy as np
import os
import pandas as pd
from keras import models, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Layer, InputSpec
from keras.layers import concatenate, Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, Dropout, BatchNormalization
from keras import Model
from keras.applications import MobileNet
from keras import backend as K

# Define the directories for training and validation data
train_dir = "C:/Users/raouf/Desktop/pfe/dataset/train"
validation_dir = "C:/Users/raouf/Desktop/pfe/dataset/val"

# Load dataset
image_dir = "C:/Users/raouf/Desktop/pfe/dataset"
#multiclass_df = pd.read_csv("/kaggle/input/balanced-datasets/Adasyn_dataset/labels.csv")

# Define binary and malignant dataframes
"""
benign_df = multiclass_df.loc[(multiclass_df['label'] == 'DF') | (multiclass_df['label'] == 'BKL') | (multiclass_df['label'] == 'NV')]
benign_df.loc[:, 'label'] = "0"
malignant_df = multiclass_df.loc[(multiclass_df['label'] == 'BCC') | (multiclass_df['label'] == 'MEL') | (multiclass_df['label'] == 'AKIEC') | (multiclass_df['label'] == 'VASC')]
malignant_df.loc[:, 'label'] = "1"
binary_df = pd.concat([benign_df, malignant_df])
"""

num_epochs = 50
mode = "binary"

if mode == "binary":
    #df = binary_df
    loss_function = "binary_crossentropy"
    num_classes = 2
else:
    #df = malignant_df
    loss_function = "categorical_crossentropy"
    num_classes = 4

from keras import backend as K
from keras.layers import Layer
import keras.layers as kl

# Soft Attention Layer Definition
class SoftAttention(Layer):
    def __init__(self, ch, m, concat_with_x=False, aggregate=False, **kwargs):
        self.channels = int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x
        super(SoftAttention, self).__init__(**kwargs)


    def build(self, input_shape):

        self.i_shape = input_shape

        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads)  # DHWC

        self.out_attention_maps_shape = input_shape[0:1] + (self.multiheads,) + input_shape[1:-1]

        if self.aggregate_channels == False:

            self.out_features_shape = input_shape[:-1] + (input_shape[-1] + (input_shape[-1] * self.multiheads),)
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1] + (input_shape[-1] * 2,)
            else:
                self.out_features_shape = input_shape

        self.kernel_conv3d = self.add_weight(shape=kernel_shape_conv3d,
                                             initializer='he_uniform',
                                             name='kernel_conv3d')
        self.bias_conv3d = self.add_weight(shape=(self.multiheads,),
                                           initializer='zeros',
                                           name='bias_conv3d')

        super(SoftAttention, self).build(input_shape)

    def call(self, x):

        exp_x = K.expand_dims(x, axis=-1)

        c3d = K.conv3d(exp_x,
                       kernel=self.kernel_conv3d,
                       strides=(1, 1, self.i_shape[-1]), padding='same', data_format='channels_last')
        conv3d = K.bias_add(c3d,
                            self.bias_conv3d)
        conv3d = kl.Activation('relu')(conv3d)

        conv3d = K.permute_dimensions(conv3d, pattern=(0, 4, 1, 2, 3))

        conv3d = K.squeeze(conv3d, axis=-1)
        conv3d = K.reshape(conv3d, shape=(-1, self.multiheads, self.i_shape[1] * self.i_shape[2]))

        softmax_alpha = K.softmax(conv3d, axis=-1)
        softmax_alpha = kl.Reshape(target_shape=(self.multiheads, self.i_shape[1], self.i_shape[2]))(softmax_alpha)

        if self.aggregate_channels == False:
            exp_softmax_alpha = K.expand_dims(softmax_alpha, axis=-1)
            exp_softmax_alpha = K.permute_dimensions(exp_softmax_alpha, pattern=(0, 2, 3, 1, 4))

            x_exp = K.expand_dims(x, axis=-2)

            u = kl.Multiply()([exp_softmax_alpha, x_exp])

            u = kl.Reshape(target_shape=(self.i_shape[1], self.i_shape[2], u.shape[-1] * u.shape[-2]))(u)

        else:
            exp_softmax_alpha = K.permute_dimensions(softmax_alpha, pattern=(0, 2, 3, 1))

            exp_softmax_alpha = K.sum(exp_softmax_alpha, axis=-1)

            exp_softmax_alpha = K.expand_dims(exp_softmax_alpha, axis=-1)

            u = kl.Multiply()([exp_softmax_alpha, x])

        if self.concat_input_with_scaled:
            o = kl.Concatenate(axis=-1)([u, x])
        else:
            o = u

        return [o, softmax_alpha]


    def compute_output_shape(self, input_shape):
        return [self.out_features_shape, self.out_attention_maps_shape]

    def get_config(self):
        return super(SoftAttention, self).get_config()



# Model Configuration

# Your code for data directories, imports, and Soft Attention layer definition

# Model Configuration
def build_model():
    # Define the base model
    mobile_net = MobileNet(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    mobile_net.trainable = False
    conv = mobile_net.layers[-6].output

    attention_layer, map2 = SoftAttention(aggregate=True, m=16, concat_with_x=False, ch=int(conv.shape[-1]), name='soft_attention')(conv)
    attention_layer = (MaxPooling2D(pool_size=(2, 2), padding="same")(attention_layer))
    conv = (MaxPooling2D(pool_size=(2, 2), padding="same")(conv))

    param=[conv, attention_layer]
    conv = concatenate(param)
    #conv = Activation("relu")(conv)
    conv = Dense(1, activation='sigmoid')(conv)

    conv = Dropout(0.2)(conv)
    conv = (Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv))
    conv = (BatchNormalization()(conv))
    conv = (Conv2D(filters=512, kernel_size=(1, 1), activation="relu", padding="same", kernel_initializer='he_normal')(conv))
    conv = (BatchNormalization()(conv))
    conv = (MaxPooling2D(pool_size=(4, 4), padding="same")(conv))
    conv = (Flatten()(conv))
    conv = (Dense(1024, activation="relu")(conv))
    conv = (Dense(num_classes, activation="softmax")(conv))

    model = Model(inputs=mobile_net.inputs, outputs=conv)
    model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])
    return model

# Define callbacks
es = EarlyStopping(monitor="val_accuracy", verbose=1, min_delta=0.01, patience=10)
mc = ModelCheckpoint(monitor="val_accuracy", verbose=1, filepath="./bestmodel.h5", save_best_only=True)
reducelr = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, patience=5, factor=0.5, min_lr=1e-7)
cb = [mc]
batch_size = 32

# Data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

# Flow validation images in batches using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

# Build and train the model
model = build_model()

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=cb
)
