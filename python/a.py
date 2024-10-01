import cv2
import dask
from dask import delayed, compute
import dask.array as da
import numpy as np
import os

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_path):
    """
    Detects faces in an image, draws a green rectangle around them, and returns the processed image.
    """
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale (necessary for Haar Cascade)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using the Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw a green rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Return the processed image
    return img

def save_image(image, output_path):
    """
    Save the image to the specified path.
    """
    cv2.imwrite(output_path, image)

def process_image(image_path, output_path):
    """
    Detects faces in the image and saves the output with rectangles drawn.
    """
    # Detect faces and draw rectangles
    img_with_faces = detect_faces(image_path)
    
    # Save the processed image
    save_image(img_with_faces, output_path)

# Example list of images to process in parallel
image_paths = ["1.jpg"]  # Add your image paths here
output_paths = ["output1.jpg", "output2.jpg", "output3.jpg"]

# Use Dask's delayed function to parallelize the image processing
tasks = [delayed(process_image)(image_path, output_path) for image_path, output_path in zip(image_paths, output_paths)]

# Execute all tasks in parallel
compute(*tasks)

print("Face detection completed. Processed images are saved.")
