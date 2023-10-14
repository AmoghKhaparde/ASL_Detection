import cv2
import os
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

def transform(image):
    # Resize the whole image to 224 by 224
    image = cv2.resize(image, (224, 224))
    
    # Random Rotation
    angle = np.random.uniform(-15, 15)
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))

    # Color Jitter (Contrast and Brightness)
    # alpha = np.random.uniform(1, 10)  # Contrast factor
    # beta = np.random.uniform(-10, 20)    # Brightness factor
    # image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Random Resized Crop
    scale = np.random.uniform(0.5, 1.0)  # Scale factor
    resized_rows, resized_cols = int(rows * scale), int(cols * scale)
    image = cv2.resize(image, (resized_cols, resized_rows))
    
    x_start = np.random.randint(0, cols - 210)
    y_start = np.random.randint(0, rows - 210)
    image = image[y_start:y_start+210, x_start:x_start+210]

    return image


num_augmentations = 10

for alphabet in 'abcdefgh' :
    
    folder_path = "new_data/" # Replace with the path to the parent folder where you want to create the new folder
    # Join the parent folder path and the new folder name to get the full path
    new_folder_path = os.path.join(folder_path, alphabet)
    # Use the os.makedirs() function to create the folder
    os.makedirs(new_folder_path)

    for num in range(1, len(os.listdir('pre_cropped_training_pics/'+alphabet+'/'))+1) :
        try:
            # Path to your input JPEG image
            image_path = 'pre_cropped_training_pics/'+alphabet+'/Photo-'+str(num)+'.jpeg'

            # Load the input image
            image = cv2.imread(image_path)

            # Convert the image to RGB format
            cv2.imshow('frame1', image)

        
            for i in range(num_augmentations):
                augmented_image = transform(image)
                # augmented_image = (np.array(augmented_image) * 255).astype(np.uint8)  # Convert to NumPy array

                cv2.imshow('frame', augmented_image)
                cv2.waitKey(1)  # Add a delay to see the image (0 means wait indefinitely)

                # Save augmented image
                base_filename, ext = os.path.splitext(os.path.basename(image_path))
                output_filename = f"{base_filename}_aug_{i}{ext}"
                output_path = os.path.join("new_data/"+alphabet+"/", output_filename)
                cv2.imwrite(output_path, augmented_image)
        except :
            continue
