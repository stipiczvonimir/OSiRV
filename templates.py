import os
import numpy as np
from mnist import load 
import cv2

X_train, Y_train, X_test, Y_test = load()


for digit in range(10):  

    digit_images = X_train[Y_train == digit][:20]  # Edit this depeninding on how many templates for each digit you want

    output_dir = f'./templates/{digit}/'
    os.makedirs(output_dir, exist_ok=True)

    for i, template in enumerate(digit_images):
        template = template.astype(np.uint8)
        output_path = os.path.join(output_dir, f"template{i+1}.png")
        cv2.imwrite(output_path, template)

        print(f"Saved {output_path}")

print("All templates saved successfully!")
