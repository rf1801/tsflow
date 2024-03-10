from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers, models
from keras.layers import concatenate, Dense, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout
from keras import *
from keras.applications import MobileNet
from keras.applications import VGG16
from keras.applications.resnet import ResNet50
from keras import backend as K
from keras.layers import Layer
import keras.layers as kl
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
import keras.applications

image_dir = "C:/Users/raouf/Desktop/pfe/dataset"
train_dir = "C:/Users/raouf/Desktop/pfe/dataset/train"
validation_dir = "C:/Users/raouf/Desktop/pfe/dataset/val"
saveHistoryDirectory='C:/Users/raouf/Desktop/pfe/history/resnet/'

#comment this if you are professor
from myDirectory import *


num_epochs = 10
mode = "binary"
loss_function = "binary_crossentropy"
num_classes = 2
batch_size = 96


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
        conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        conv_base.trainable = False
    return conv_base





def build_model(arch):
    from_scratch = choose_conv_base(arch)
    conv = from_scratch.layers[-1].output

    #attention
    #attention_layer, map2 = SoftAttention(aggregate=True, m=16, concat_with_x=False, ch=int(conv.shape[-1]),name='soft_attention')(conv)
    #attention_layer = (MaxPooling2D(pool_size=(2, 2), padding="same")(attention_layer))



    conv = (MaxPooling2D(pool_size=(2, 2), padding="same")(conv))


    #attention
    #param=[conv, attention_layer]
    param = [conv]

    conv = concatenate(param)
    conv = Activation("relu")(conv)
    conv = Dropout(0.2)(conv)
    conv = (Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv))
    conv = (BatchNormalization()(conv))
    conv = (Conv2D(filters=512, kernel_size=(1, 1), activation="relu", padding="same", kernel_initializer='he_normal')(conv))
    conv = (BatchNormalization()(conv))
    conv = (MaxPooling2D(pool_size=(4, 4), padding="same")(conv))
    conv = (Flatten()(conv))
    conv = (Dense(1024, activation="relu")(conv))
    conv = (Dense(2, activation="softmax")(conv))
    model = Model(inputs=from_scratch.inputs, outputs=conv)
    # model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005), loss=loss_function, metrics=["accuracy"])
    return model


# adding some callbacks : early stopping, modelCheckpoint, and reduce learning rate on plateau
es = EarlyStopping(monitor="val_accuracy", verbose=1, min_delta=0.01, patience=6)
mc = ModelCheckpoint(monitor="val_accuracy", verbose=1, filepath="./bestmodel.h5", save_best_only=True)
reducelr = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, patience=2, factor=0.9, min_lr=1e-7)

# put callbacks to use in the cb array
cb = [mc,reducelr,es]




# Set up data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255 , rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(224, 224),batch_size=batch_size,class_mode='categorical')


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')





import time

architecture="resnet"


start_time = time.time()


# Compile and train the model
model = build_model(architecture)
history = model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size,
                              epochs=num_epochs, validation_data=validation_generator,
                              validation_steps=validation_generator.samples // batch_size, callbacks=cb)
history_df = pd.DataFrame(history.history)
end_time = time.time()
duration = round(end_time - start_time)
print("Execution time:", duration, "seconds")


# Save the DataFrame to an Excel file

history_excel_path = saveHistoryDirectory +architecture+ '_history_'+ str(duration)+  '.xlsx'
history_df.to_excel(history_excel_path, index=False)
print("Training history saved to:", history_excel_path)
