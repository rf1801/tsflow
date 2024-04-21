import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input



model_path="C:/Users/raouf/Desktop/pfe/history/no_overfit/mobilenet_epochs_3_batch_64_lr_0.001_attention_False.keras"
folder_path = "C:/Users/raouf/Desktop/pfe/binary_mura/val/not fractured/"


# Load the image and resize it to the target size
def preprocess_image(image_path, target_size=(224, 224), model_name="mobilenet"):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create a batch
    #img_array = img_array / 255.0
    return img_array




model = load_model(model_path)  # Replace 'path_to_your_model.h5' with the actual path

if "mobilenet" in model_path:
    my_model = "mobilenet"
elif "vgg16" in model_path:
    my_model = "vgg16"
elif "resnet" in model_path:
    my_model = "resnet"
elif "from_scratch" in model_path:
    my_model = "from_scratch"
else:
    raise ValueError("Unknown model type")

# Define the folder containing the Excel files


# Get a list of all files in the specified directory
files = [file for file in os.listdir(folder_path)]

frac=0
not_frac= 0

# Iterate over each file
for file_name in files:
    image_path = os.path.join(folder_path, file_name)
    test_image = preprocess_image(image_path, model_name=my_model)
    class_names = ['fractured', 'not fractured']  # Replace with your actual class names
    result=class_names[np.argmax(model.predict(test_image))]
    print(round(100 * (frac + not_frac) / len(files), 2), " % Done")
    #print("Predicted class:",  result )

    if result=="fractured":
        frac=frac+1
    else:
        not_frac=not_frac+1



print("fractured     :",frac,"occurances , percentage :",100*frac/len(files))
print("not fractured :",not_frac,"occurances , percentage :",100*not_frac/len(files))



