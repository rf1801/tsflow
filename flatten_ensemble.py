import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Concatenate, Flatten, Dense
from tensorflow.keras.applications import MobileNet, VGG16

#image_dir = "C:/Users/raouf/Desktop/pfe/my_dataset"
train_dir = "C:/Users/raouf/Desktop/pfe/my_dataset/train"
test_dir = "C:/Users/raouf/Desktop/pfe/my_dataset/test"
validation_dir = "C:/Users/raouf/Desktop/pfe/my_dataset/val"



# Define constants
IMAGE_SIZE = (224,224)
BATCH_SIZE = 64
NUM_CLASSES = 2
EPOCHS = 10

# Load pretrained models
mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Set layers as non-trainable
for layer in mobilenet_model.layers:
    layer.trainable = False
for layer in vgg16_model.layers:
    layer.trainable = False

for layer in vgg16_model.layers:
    print(layer)


mobilenet_output = Flatten()(mobilenet_model.output)
# Flatten the output of the VGG16 model
vgg16_output = Flatten()(vgg16_model.output)

# Concatenate the flattened outputs
concatenated = Concatenate()([mobilenet_output, vgg16_output])


# Add dense layers for classification
x = Dense(128, activation='relu')(concatenated)
output = Dense(2, activation='softmax')(x)

# Create the model
model = Model(inputs=[mobilenet_model.input, vgg16_model.input], outputs=output)




















# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Create data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    #subset='training'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Train the model
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Save the model
model.save('ensemble_model.h5')
