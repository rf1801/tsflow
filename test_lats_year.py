import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras import layers, models

"""# Importing dataset:"""

image_dir = "C:/Users/raouf/Desktop/pfe/dataset"

"""
multiclass_df = pd.read_csv("/kaggle/input/balanced-datasets/Adasyn_dataset/labels.csv")

benign_df = multiclass_df.loc[
    (multiclass_df['label'] == 'DF') | (multiclass_df['label'] == 'BKL') | (multiclass_df['label'] == 'NV')]
benign_df.loc[:, 'label'] = "0"
malignant_df = multiclass_df.loc[
    (multiclass_df['label'] == 'BCC') | (multiclass_df['label'] == 'MEL') | (multiclass_df['label'] == 'AKIEC') | (
                multiclass_df['label'] == 'VASC')]
malignant_df.loc[:, 'label'] = "1"
binary_df = pd.concat([benign_df, malignant_df])
"""

num_epochs = 50
# mode = "binary"
mode = "binary"

if (mode == "binary"):
    #df = binary_df
    loss_function = "binary_crossentropy"
    num_classes = 2
else:
    #df = malignant_df
    loss_function = "categorical_crossentropy"
    num_classes = 4

"""# Soft Attentionn"""

# Soft Attention

from keras import backend as K
from keras.layers import Layer
import keras.layers as kl


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


"""# configuring model:"""

from keras.layers import concatenate, Dense, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout
from keras import *
from keras.applications import MobileNet
from keras.applications import VGG16
from keras.applications.resnet import ResNet50
import tensorflow as tf



#from tensorflow.keras.applications import EfficientNetB0


def choose_conv_base(name="from_scratch", input_shape=(224, 224, 3)):
    if (name == "from_scratch"):
        conv_base = models.Sequential()
        conv_base.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        conv_base.add(layers.MaxPooling2D((2, 2)))
        conv_base.add(layers.Conv2D(64, (3, 3), activation='relu'))
        conv_base.add(layers.MaxPooling2D((2, 2)))
        conv_base.add(layers.Conv2D(128, (3, 3), activation='relu'))
        conv_base.add(layers.MaxPooling2D((2, 2)))
        conv_base.add(layers.Conv2D(128, (3, 3), activation='relu'))
        conv_base.add(layers.MaxPooling2D((2, 2)))
    elif (name == "mobilenet"):
        conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
        conv_base.trainable = False
    elif (name == "vgg16"):
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        conv_base.trainable = False
    elif (name == "resnet"):
        conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        conv_base.trainable = False
    elif (name == "efficientnet"):
        #conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        #conv_base.trainable = False
        print("test test")
    return conv_base


def build_model():
    mobile_net = MobileNet(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    mobile_net.trainable = False
    #conv = mobile_net.layers[-6].output

    # overwritting with from scratch
    from_scratch = choose_conv_base("from_scratch")
    conv = from_scratch.layers[-1].output
    # Add the Soft Attention Layer


    attention_layer, map2 = SoftAttention(aggregate=True, m=16, concat_with_x=False, ch=int(conv.shape[-1]), name='soft_attention')(conv)
    attention_layer = (MaxPooling2D(pool_size=(2, 2), padding="same")(attention_layer))
    conv = (MaxPooling2D(pool_size=(2, 2), padding="same")(conv))

    param=[conv, attention_layer]
    conv = concatenate(param)
    conv = Activation("relu")(conv)
    conv = Dropout(0.2)(conv)
    conv = (Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(
        conv))
    conv = (BatchNormalization()(conv))
    conv = (Conv2D(filters=512, kernel_size=(1, 1), activation="relu", padding="same", kernel_initializer='he_normal')(
        conv))
    conv = (BatchNormalization()(conv))
    conv = (MaxPooling2D(pool_size=(4, 4), padding="same")(conv))
    conv = (Flatten()(conv))
    conv = (Dense(1024, activation="relu")(conv))
    conv = (Dense(7, activation="softmax")(conv))

    model = Model(inputs=mobile_net.inputs, outputs=conv)
    # model.summary()
    model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])
    return model


# adding some callbacks : early stopping, modelCheckpoint, and reduce learning rate on plateau
es = EarlyStopping(monitor="val_accuracy", verbose=1, min_delta=0.01, patience=10)
mc = ModelCheckpoint(monitor="val_accuracy", verbose=1,
                     filepath="./bestmodel.h5", save_best_only=True)
reducelr = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, patience=5, factor=0.5, min_lr=1e-7)
# put callbacks to use in the cb array
cb = [mc]
batch_size = 32



# Load and preprocess your dataset
# Example:
train_dir = "C:/Users/raouf/Desktop/pfe/dataset/train"
validation_dir = "C:/Users/raouf/Desktop/pfe/dataset/val"

# Set up data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'  # or 'binary' depending on your mode
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'  # or 'binary' depending on your mode
)

# Compile and train the model
model = build_model()

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=cb
)
