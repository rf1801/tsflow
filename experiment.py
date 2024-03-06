import time
import os
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

# Define the directory containing your dataset
image_dir = "C:/Users/raouf/Desktop/pfe/dataset"
train_dir = os.path.join(image_dir, "train")
validation_dir = os.path.join(image_dir, "val")

num_epochs = 10
mode = "binary"
loss_function = "binary_crossentropy"
num_classes = 2
batch_size = 32

# Set up data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')
validation_generator = train_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')

# Define the architecture you want to use
architecture = "efficientnet"

def build_model(arch):
    from_scratch = choose_conv_base(arch)
    conv = from_scratch.layers[-1].output

    #attention
    attention_layer, map2 = SoftAttention(aggregate=True, m=16, concat_with_x=False, ch=int(conv.shape[-1]),name='soft_attention')(conv)
    attention_layer = (MaxPooling2D(pool_size=(2, 2), padding="same")(attention_layer))



    conv = (MaxPooling2D(pool_size=(2, 2), padding="same")(conv))


    #attention
    param=[conv, attention_layer]


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
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=loss_function, metrics=["accuracy"])
    return model

# Start the loop to adjust learning rate
learning_rates = [0.001, 0.0001, 0.00001]  # Example learning rates to try

for lr in learning_rates:
    start_time = time.time()

    # Compile and train the model with the current learning rate
    model = build_model(architecture)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])

    # Define callbacks
    es = EarlyStopping(monitor="val_accuracy", verbose=1, min_delta=0.01, patience=10)
    mc = ModelCheckpoint(monitor="val_accuracy", verbose=1, filepath="./bestmodel.h5", save_best_only=True)
    reducelr = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, patience=2, factor=0.1, min_lr=1e-7)
    cb = [mc, reducelr]

    # Train the model
    history = model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size,
                                  epochs=num_epochs, validation_data=validation_generator,
                                  validation_steps=validation_generator.samples // batch_size, callbacks=cb)

    # Save the training history to an Excel file
    history_df = pd.DataFrame(history.history)
    duration = round(time.time() - start_time)
    history_excel_path = f'C:/Users/raouf/Desktop/pfe/history/to compare effnet soft/{architecture}_soft_attention_history_lr{lr}_{duration}.xlsx'
    history_df.to_excel(history_excel_path, index=False)
    print("Training history saved to:", history_excel_path)

    print(f"Execution time for learning rate {lr}: {duration} seconds\n")
