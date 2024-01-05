import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_holes(my_image):
    """
    Removes holes from a binary image.

    Parameters:
    my_image (numpy.ndarray): A binary image represented as a 2D numpy array.

    Returns:
    numpy.ndarray: The input image with holes removed.
    """
    
    # TODO: implement the function
    
    my_image_neg = np.logical_not(my_image).astype(np.uint8)
    
    #Returns Connected Components count and labels representing the different connected components, label 0 = background
    ccCount, labels = cv2.connectedComponents(my_image_neg, connectivity=4)
    
    #Returns largest label of largest connected component and its size
    #CC_STAT_AREA = a number representing the area of connected component
    #key=lamba x: x[1] = returns the second element since the background is the largest connected component
    max_label, max_size = max([(i, labels[i, cv2.CC_STAT_AREA]) for i in range(1,ccCount)], key=lambda x: x[1])
    
    # Changes values of connected components to 0
    #label cannot equal max_label, which is the background
    my_image_neg[labels != max_label] = 0
    
    #Take the negation of the image to change it back to normal
    result = np.logical_not(my_image_neg).astype(np.uint8)
    
    return result
