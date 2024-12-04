# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:16:24 2024

@author: greeshma-ts283
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
# Load the image
image_path = 'D:/Manuscript_WORK_ZOHO/first_paragarpgh_BL54_002/table1.png'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding for better contour detection
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Use morphological operations to clean up the noise
#kernel = np.ones((3, 3), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)


# Detect edges using the Canny edge detector
edges = cv2.Canny(binary, 50, 150, apertureSize=3)

# Use HoughLinesP to detect lines in the image
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=15)

# Draw lines on the binary image to help isolate table cells
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(binary, (x1, y1), (x2, y2), (0, 0, 0), 3)

# Find contours on the processed binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours from top to bottom, left to right
contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[1], cv2.boundingRect(ctr)[0]))

# Create a directory to save extracted characters
output_dir = 'D:/Manuscript_WORK_ZOHO/first_paragarpgh_BL54_002/extracted_characters'
os.makedirs(output_dir, exist_ok=True)

# Initialize counters
valid_chars_count = 0
min_char_width, min_char_height = 12, 12  # Minimum size of a valid character
max_chars_to_extract = 182 # The number of Tamil characters to extract

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    
    # Filter out small contours that are unlikely to be characters
    if w > min_char_width and h > min_char_height:
        # Increase bounding box slightly to capture full character
        padding = 5
        x, y = max(x - padding, 0), max(y - padding, 0)
        w, h = min(w + 2 * padding, image.shape[1] - x), min(h + 2 * padding, image.shape[0] - y)
        
        # Crop the character from the original image
        char_image = image[y:y+h, x:x+w]
        
        # Save the character as an image file
        if valid_chars_count < max_chars_to_extract:
            char_image_path = os.path.join(output_dir, f'char_{valid_chars_count}.png')
            cv2.imwrite(char_image_path, char_image)
            valid_chars_count += 1

# Check if the number of extracted characters matches the expected 187
if valid_chars_count == max_chars_to_extract:
    print(f'Successfully extracted {valid_chars_count} Tamil characters.')
else:
    print(f'Extracted {valid_chars_count} characters, but expected 187.')
