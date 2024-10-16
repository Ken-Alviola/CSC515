#!/usr/bin/env python
# coding: utf-


import cv2
import matplotlib.pyplot as plt


image_list = ['indoor.jpg','outdoor.jpg','closeup.jpg',]


for image in image_list:
    # Load the image in grayscale
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian Blur to smooth the image and reduce noise
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # 1. Otsu's Binarization
    _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Adaptive Thresholding (Gaussian and Mean)
    adaptive_thresh_mean = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                                 cv2.THRESH_BINARY_INV, 11, 2)
    adaptive_thresh_gaussian = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                     cv2.THRESH_BINARY_INV, 11, 2)

    # Plotting the results side by side
    plt.figure(figsize=(10, 8))

    # Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Otsu's Binarization
    plt.subplot(2, 2, 2)
    plt.imshow(otsu_thresh, cmap='gray')
    plt.title("Otsu's Binarization")
    plt.axis('off')

    # Adaptive Thresholding (Mean)
    plt.subplot(2, 2, 3)
    plt.imshow(adaptive_thresh_mean, cmap='gray')
    plt.title('Adaptive Thresholding (Mean)')
    plt.axis('off')

    # Adaptive Thresholding (Gaussian)
    plt.subplot(2, 2, 4)
    plt.imshow(adaptive_thresh_gaussian, cmap='gray')
    plt.title('Adaptive Thresholding (Gaussian)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
