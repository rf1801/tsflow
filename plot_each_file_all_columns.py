import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folder containing the Excel files
folder_path = 'C:/Users/raouf/Desktop/pfe/history/'

# Get a list of all files in the specified directory
files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]

# Iterate over each file
for file_name in files:
    # Construct the full path to the file
    file_path = os.path.join(folder_path, file_name)

    # Read data from Excel file
    data = pd.read_excel(file_path)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot loss
    plt.plot(data['loss'], label='Loss', color='blue')

    # Plot accuracy
    plt.plot(data['accuracy'], label='Accuracy', color='green')

    # Plot val_loss
    plt.plot(data['val_loss'], label='Validation Loss', color='red')

    # Plot val_accuracy
    plt.plot(data['val_accuracy'], label='Validation Accuracy', color='orange')

    # Set title and labels
    plt.title(f'Plot for {file_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Value')

    # Show legend
    plt.legend()

    # Show plot
    plt.show()
