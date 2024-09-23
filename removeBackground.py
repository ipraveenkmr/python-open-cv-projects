import cv2
import numpy as np
from rembg import remove

# Load the image from file
input_path = 'my.jpg'  # Specify the path to your input image
output_path = 'output_image.png'  # The output will be a PNG with transparent background

# Read the image
image = cv2.imread(input_path)

# Remove the background
result = remove(image)

# Save the output image
cv2.imwrite(output_path, result)

print(f"Background removed! Saved the result to {output_path}")
