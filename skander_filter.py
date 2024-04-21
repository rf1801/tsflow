from imports import *

from myDirectory import *

num_epochs = 3
mode = "binary"
loss_function = "binary_crossentropy"
num_classes = 2
batch_size = 64
learning_rate = 0.001
soft_attention_enabled = False
architecture="vgg16"


# Define the preprocess_image function
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Reduce brightness
    brightness_factor = 0.6  # Adjust this value to reduce brightness (0.0 < brightness_factor < 1.0)
    darkened_image = cv2.convertScaleAbs(image * brightness_factor)

    # Apply adaptive histogram equalization with increased contrast
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))  # Increase clipLimit for more contrast
    equalized_image = clahe.apply(darkened_image)

    return equalized_image

# Set up data generators with preprocessing function
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255
)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')


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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_function, metrics=["accuracy"])
    return model


model_name = (
        architecture + '_epochs_' + str(num_epochs) +
        '_batch_' + str(batch_size) +
        '_lr_' + str(learning_rate) +  # Assuming fixed learning rate
        '_attention_' + str(soft_attention_enabled) +  # Assuming soft attention is disabled

        '.keras'
)

# adding some callbacks : early stopping, modelCheckpoint, and reduce learning rate on plateau
es = EarlyStopping(monitor="val_accuracy", verbose=1, min_delta=0.01, patience=6)
mc = ModelCheckpoint(monitor="val_accuracy", verbose=1, filepath=saveHistoryDirectory + model_name,
                     save_best_only=True)
reducelr = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, patience=1, factor=0.9, min_lr=1e-7)

# put callbacks to use in the cb array
cb = [mc,reducelr,es]





import time




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

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Save the DataFrame to an Excel file

# Extracting metric values
loss = history_df['loss'].iloc[
    -1]  # Final training loss
accuracy = history_df['accuracy'].iloc[
    -1]  # Final training accuracy
val_loss = history_df['val_loss'].iloc[
    -1]  # Final validation loss
val_accuracy = history_df['val_accuracy'].iloc[
    -1]  # Final validation accuracy

# Constructing the file name with metric values
history_excel_path = (
        saveHistoryDirectory +
        architecture + '_history_epochs_' + str(num_epochs) +
        '_batch_' + str(batch_size) +
        '_lr_' + str(
    learning_rate) +  # Assuming fixed learning rate
        '_attention_' + str(
    False) +  # Assuming soft attention is disabled
        '_metrics_' + f'loss_{round(loss, 2)}_accuracy_{round(accuracy, 2)}_val_loss_{round(val_loss, 2)}_val_accuracy_{round(val_accuracy, 2)}' +
        '_test_loss_' + str(round(test_loss, 2)) +
        '_test_accuracy_' + str(round(test_accuracy, 2)) +
        '_duration_' + str(duration) +
        '.xlsx'
)

history_df.to_excel(history_excel_path, index=False)
print("Training history saved to:", history_excel_path)
