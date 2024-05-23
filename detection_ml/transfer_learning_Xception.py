import numpy as np
import os
from sklearn.model_selection import train_test_split
import overfit
from matplotlib import pyplot as plt
from keras.applications import Xception
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# Assuming you have a list of file paths for images and corresponding labels
# Replace this with your actual code to load file paths and labels
# For demonstration purposes, let's assume you have lists of file paths and labels
# Path to the folder containing the images
folder_path = r"C:\Users\Hugo Greulich Mayor\Desktop\EPFL\MA2\SWISSCAT\Photos ML\data_img_Total"

# Initialize lists to store file paths and labels
file_paths = []
labels = []

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith("_soluble.jpg"):
        # Add file path with prefix and label 0 for soluble
        file_paths.append(os.path.join(folder_path, filename))
        labels.append(0)
    elif filename.endswith("_insoluble.jpg"):
        # Add file path with prefix and label 1 for insoluble
        file_paths.append(os.path.join(folder_path, filename))
        labels.append(1)

# Load and preprocess images
X = []
for file_path in file_paths:
    img = load_img(file_path, target_size=(299, 299))
    img_array = img_to_array(img)
    X.append(img_array)
X = np.array(X)
X = X.astype('float32') / 255.0  # Normalize pixel values to [0, 1]

# Convert labels to one-hot encoding
y = to_categorical(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_samples = len(y_test)
num_soluble = np.sum(y_test == 0)
num_insoluble = np.sum(y_test == 1)

num_samples_train = len(y_train)
num_soluble_train = np.sum(y_train == 0)
num_insoluble_train = np.sum(y_train == 1)
# Calculate the baseline accuracy
baseline_accuracy = max(num_soluble_train, num_insoluble_train) / num_samples_train

print("Baseline Accuracy:", baseline_accuracy)

# Load pre-trained Xception model without top layer
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Add custom classification layers on top of the base model
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(2, activation='softmax'))  # Assuming 2 classes (dissolved and not dissolved)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot training history
overfit.plot_training_history(history)

model.save("detection_Xception_e_100_bs_64.keras")