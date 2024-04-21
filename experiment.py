from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers, models
from keras.layers import concatenate, Dense, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout, \
    Attention
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
import time


tf.keras.backend.clear_session()

def saveHistory():

    # Save the DataFrame to an Excel file

    # Extracting metric values
    loss = history_df['loss'].iloc[-1]  # Final training loss
    accuracy = history_df['accuracy'].iloc[-1]  # Final training accuracy
    val_loss = history_df['val_loss'].iloc[-1]  # Final validation loss
    val_accuracy = history_df['val_accuracy'].iloc[-1]  # Final validation accuracy

    # Constructing the file name with metric values
    history_excel_path = (
            saveHistoryDirectory +
            architecture + '_history_epochs_' + str(num_epochs) +
            '_batch_' + str(batch_size) +
            '_lr_' + str(learning_rate) +  # Assuming fixed learning rate
            '_attention_' + str(soft_attention_enabled) +  # Assuming soft attention is disabled
            '_metrics_' + f'loss_{round(loss, 2)}_accuracy_{round(accuracy, 2)}_val_loss_{round(val_loss, 2)}_val_accuracy_{round(val_accuracy, 2)}' +
            '_test_loss_' + str(round(test_loss, 2)) +
            '_test_accuracy_' + str(round(test_accuracy, 2)) +
            '_duration_' + str(duration) +
            '.xlsx'
    )

    history_df.to_excel(history_excel_path, index=False)
    print("Training history saved to:", history_excel_path)


def saveModel():
    name = (architecture + '_epochs_' + str(num_epochs) +
            '_batch_' + str(batch_size) +
            '_lr_' + str(learning_rate) +
            '_attention_' + str(soft_attention_enabled) +
            '.keras')
    return name


# comment this if you are professor
from myDirectory import *

# ,"resnet","mobilenet","from_scratch"

architecture="mobilenet"
architectures = ["mobilenet","vgg16"]
batch_size = 64
batches=[8,16,32,96,128,256]
num_epochs = 5
mode = "binary"
loss_function = "binary_crossentropy"
num_classes = 2
learning_rate = 0.005
soft_attention_enabled = False


def choose_conv_base(name="from_scratch", input_shape=(224, 224, 3)):
    if name == "from_scratch":
        conv_base = models.Sequential()
        conv_base.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        conv_base.add(layers.MaxPooling2D((2, 2)))
        conv_base.add(layers.Conv2D(64, (3, 3), activation='relu'))
        conv_base.add(layers.MaxPooling2D((2, 2)))
        conv_base.add(layers.Conv2D(128, (3, 3), activation='relu'))
        conv_base.add(layers.MaxPooling2D((2, 2)))
        conv_base.add(layers.Conv2D(128, (3, 3), activation='relu'))
        conv_base.add(layers.MaxPooling2D((2, 2)))
    elif name == "mobilenet":
        conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
        conv_base.trainable = False
    elif name == "vgg16":
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        conv_base.trainable = False
    elif name == "resnet":
        conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        conv_base.trainable = False
    elif name == "efficientnet":
        conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        conv_base.trainable = False
    return conv_base

class SoftAttention(Layer):
    def __init__(self):
        super(SoftAttention, self).__init__()

    def build(self, input_shape):
        self.context_vector = self.add_weight(shape=(input_shape[-1], 1),
                                               initializer='glorot_uniform',
                                               trainable=True,
                                               name='context_vector')
        super(SoftAttention, self).build(input_shape)

    def call(self, inputs):
        # Compute attention scores
        attention_weights = tf.nn.softmax(tf.matmul(inputs, self.context_vector), axis=1)
        # Apply attention weights to inputs
        weighted_sum = tf.reduce_sum(inputs * attention_weights, axis=1)
        return weighted_sum

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def build_model(arch):
    from_scratch = choose_conv_base(arch)
    conv = from_scratch.layers[-1].output
    if soft_attention_enabled:
        # Add soft attention layer
        conv = SoftAttention()(conv)
    conv = (Flatten()(conv))
    conv = (Dense(1024, activation="relu")(conv))
    conv = (Dense(2, activation="softmax")(conv))
    model = Model(inputs=from_scratch.inputs, outputs=conv)
    # model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_function, metrics=["accuracy"])
    return model


# Set up data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=60,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                                                horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=batch_size,
                                                    class_mode='categorical')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=batch_size,
                                                  class_mode='categorical')

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')



for architecture in architectures:
    model_name = saveModel()

    # adding some callbacks : early stopping, modelCheckpoint, and reduce learning rate on plateau
    es = EarlyStopping(monitor="val_accuracy", verbose=1, min_delta=0.01, patience=6)
    mc = ModelCheckpoint(monitor="val_accuracy", verbose=1, filepath=saveHistoryDirectory + model_name,save_best_only=True)
    reducelr = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, patience=1, factor=0.9, min_lr=1e-7)
    cb = [mc,
          reducelr, es]

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

    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')


    saveHistory()
