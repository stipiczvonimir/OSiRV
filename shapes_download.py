import kagglehub
import shutil
import os

# Download the dataset (downloads to default kagglehub directory)
default_path = kagglehub.dataset_download("smeschke/four-shapes")

# Define your custom path
custom_path = "C:/Users/zvono/Desktop/OSiRV/OSiRV/four_shapes"

# Move dataset to custom path
shutil.move(default_path, custom_path)

print("Dataset moved to:", custom_path)
