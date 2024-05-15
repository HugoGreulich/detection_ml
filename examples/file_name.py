import os

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

# Verify if the lengths match
print("Number of images:", len(file_paths))
print(file_paths)
print("Number of labels:", len(labels))
print(labels)
