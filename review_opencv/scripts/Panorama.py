import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Set base image path and filename pattern
base_image_path = 'review_opencv/images/'
image_prefix = 'boat'
image_suffix = '.jpg'

# Initialize list to store images
images_rgb = []
image_index = 1 # Starting image number

# Loop to read images and convert colors
while True:
    # Create file path
    image_filename = f"{image_prefix}{image_index}{image_suffix}"
    image_path = os.path.join(base_image_path, image_filename)

    # Optional: Check if file exists first (can be clearer before cv2.imread)
    # if not os.path.exists(image_path):
    #     print(f"Info: File {image_path} not found. Stopping image loading.")
    #     break

    # Read image
    img_bgr = cv2.imread(image_path)

    # Check if the image was read successfully (cv2.imread returns None if file doesn't exist or is not a valid image)
    if img_bgr is None:
        if image_index == 1: # Case: the very first image is not found
            print(f"Error: Could not find the starting image: {image_path}")
        else:
            # Case: a subsequent image is not found
            print(f"Info: Could not find the next image ({image_path}). Stopping loading. Loaded {len(images_rgb)} images in total.")
        break # Break the loop
    else:
        # Convert color space from BGR (OpenCV default) to RGB (Matplotlib default)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        images_rgb.append(img_rgb)
        image_index += 1 # Increment for the next image number

# Display images using Matplotlib
if not images_rgb:
    print("No images to display.")
else:
    num_loaded_images = len(images_rgb)
    
    # Configure subplots (fix number of columns and calculate rows)
    num_cols = 3
    # Calculate number of rows based on the number of images (ceiling division)
    num_rows = (num_loaded_images + num_cols - 1) // num_cols

    # Set figure size (can be dynamically adjusted based on image count, fixed values used here as an example)
    # For example, set height per row to 5, and width per column relative to existing figsize
    fig_height_per_row = 5
    # Try to maintain aspect ratio from original figsize=(40,10) for 2x3 grid.
    # Original width per column was 40/3.
    fig_width_per_column = 40 / 3

    fig_height = num_rows * fig_height_per_row
    fig_width = num_cols * fig_width_per_column
    
    # Prevent height from being zero if num_rows is 0 (though 'if not images_rgb' handles this)
    plt.figure(figsize=(fig_width, fig_height if fig_height > 0 else fig_height_per_row))

    for i, img_display in enumerate(images_rgb):
        plt.subplot(num_rows, num_cols, i + 1) # Subplot position (1-indexed)
        plt.imshow(img_display)
        plt.axis('off') # Hide axis information

    plt.tight_layout() # Automatically adjust subplot spacing
    plt.show()
    stitcher = cv2.Stitcher_create()
    status, panorama = stitcher.stitch(images_rgb)
    
    if status ==0:
        plt.figure(figsize=(40,10))
        plt.imshow(panorama)
        plt.axis('off')
        plt.title('Panorama')
        plt.show()