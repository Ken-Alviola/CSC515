#!/usr/bin/env python
# coding: utf-8

import cv2
import keras_ocr
import numpy as np

# Initialize Keras-OCR pipeline for detection and recognition
pipeline = keras_ocr.pipeline.Pipeline()

# Load images (ensure the paths are correct)
images = [
    cv2.imread("close.jpg"),
    cv2.imread("far.jpg"),
    cv2.imread("non_russian.jpg")
]

# Load the Haar Cascade for Russian license plates
plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
if plate_cascade.empty():
    print("Error: Could not load cascade classifier.")

# Function to preprocess using CLAHE and detect plates
def detect_license_plate(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    plates = cascade.detectMultiScale(enhanced_gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 40))
    cropped_plates = []
    
    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped_plates.append(image[y:y + h, x:x + w])
    
    return image, cropped_plates

# Detect plates
detected_images = []
all_cropped_plates = []
for img in images:
    if img is not None:
        result_img, plates = detect_license_plate(img, plate_cascade)
        detected_images.append(result_img)
        if plates:
            all_cropped_plates.extend(plates)

# Display detected plates with bounding boxes
for i, detected_img in enumerate(detected_images):
    cv2.imshow(f"Detected License Plate Image {i+1}", detected_img)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
# Display cropped plate regions (optional)
for i, plate in enumerate(all_cropped_plates):
    if plate is not None:
        cv2.imshow(f"Cropped Plate {i+1}", plate)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# Loop through all cropped plates and apply different morphological operations
for i, plate_img in enumerate(all_cropped_plates):
    if plate_img is not None:
        # Convert the image to grayscale if not already
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        plate_img = clahe.apply(plate_img)
        
        # Resize the image for consistency
        plate_img = cv2.resize(plate_img, (350, 100))
        
        # Apply a specific morphological operation based on the index
        if i == 0:          
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            processed_img = cv2.morphologyEx(plate_img, cv2.MORPH_OPEN, kernel)  
            operation = "Opening"
        elif i == 1:
            #Create the sharpening kernel
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
            # Sharpen the image 
            processed_img = cv2.filter2D(plate_img, -1, kernel)       
            # Median blur for noise reduction
            processed_img = cv2.medianBlur(processed_img,5)
            operation = "Sharpen then blur"
        elif i == 2:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            processed_img = cv2.morphologyEx(plate_img, cv2.MORPH_ERODE, kernel)
            processed_img = cv2.medianBlur(processed_img,5)
            operation = "Erode and Blur"
        elif i == 3:
            processed_img = plate_img
            #Create the sharpening kernel
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
            # Sharpen the image 
            processed_img = cv2.filter2D(processed_img, -1, kernel) 
            processed_img = cv2.medianBlur(processed_img,3)
            operation = "Sharpen then Blur"
        else:
            processed_img = plate_img  # If more images, no change

        # Convert the processed plate image to RGB for Keras-OCR
        plate_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        
        # Perform OCR using Keras-OCR pipeline
        prediction_groups = pipeline.recognize([plate_img_rgb])
        
        print()
        # Print detected text
        print(f"License Plate Text for Plate {i+1} ({operation}):")
        for text, box in prediction_groups[0]:
            print(f"Detected text: {text}")
            
        print()
        # Display the plate image with applied morphological operation (optional)
        cv2.imshow(f"License Plate {i+1} - {operation}", processed_img)
        cv2.waitKey(0)

cv2.destroyAllWindows()



