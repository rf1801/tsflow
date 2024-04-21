from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers, models
from keras.layers import concatenate, Dense, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout, Attention
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
import matplotlib.pyplot as plt
tf.keras.backend.clear_session()

num_epochs = 3
mode = "binary"
loss_function = "binary_crossentropy"
num_classes = 2
batch_size = 64
learning_rate=0.001
soft_attention_enabled = False
test_dir = "C:/Users/raouf/Desktop/pfe/my_dataset/test"


from tensorflow.keras.preprocessing import image
test_datagen = image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(224, 224),
                                                  batch_size=1,
                                                  class_mode='categorical',
                                                  shuffle=False)


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("C:/Users/raouf/Desktop/pfe/history/no_overfit/mobilenet_epochs_3_batch_64_lr_0.001_attention_False.keras")  # Replace 'path_to_your_model.h5' with the actual path


model1 = load_model("C:/Users/raouf/Desktop/pfe/history/no_overfit/mobilenet_epochs_3_batch_64_lr_0.001_attention_False.keras")
model2 = load_model("C:/Users/raouf/Desktop/pfe/history/no_overfit/vgg16_epochs_3_batch_64_lr_0.001_attention_False.keras")

# Load your test data (replace test_data with your actual test dataset)
test_data = "C:/Users/raouf/Desktop/pfe/my_dataset/test"

# Make predictions using the loaded models
predictions1 = model1.predict(test_generator)
predictions2 = model2.predict(test_generator)

# Combine predictions (soft voting)
ensemble_predictions = 0.5 * (predictions1 + predictions2)

# Threshold the ensemble predictions for binary classification
predictions = (ensemble_predictions > 0.5).astype(int)

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels from test data generator
true_classes = test_generator.classes

# Get class labels
class_labels = list(test_generator.class_indices.keys())

# Generate classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)

print(report)


from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix,
                                            display_labels = class_labels)
cm_display.plot()
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


