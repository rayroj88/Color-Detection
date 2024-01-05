import cv2
import numpy as np
import matplotlib.pyplot as plt
from get_component import get_component
from remove_holes import remove_holes

def detect_players(image_path):
    """
    Detects the positions of red and blue players in a soccer field image.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        Tuple of five elements:
        - red_player_centroids (List[Tuple[int, int]]): The (x, y) coordinates of the centroids of the detected red players.
        - blue_player_centroids (List[Tuple[int, int]]): The (x, y) coordinates of the centroids of the detected blue players.
        - field (np.ndarray): The binary image of the soccer field.
        - red_players (np.ndarray): The binary image of the detected red players.
        - blue_players (np.ndarray): The binary image of the detected blue players.
    """
    # TODO: implement the function
    
    #Set image as color
    color = cv2.imread('data/soccer_field4.jpg')
    
    color = cv2.threshold(color, 127, 255, cv2.THRESH_BINARY)[1]
    
    #change from BGR to RGB
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    
    #Separate the color bands
    r = color[:,:,0]
    g = color[:,:,1]
    b = color[:,:,2]
    
    #Identify green areas
    green = (g - r > 40) & (g - b > 72)
    green = green.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    green = cv2.erode(green, kernel)
    kernel = np.ones((7,7), np.uint8)
    green = cv2.dilate(green, kernel)
    
    
    #Remove holes to isolate field from players
    field = remove_holes(green)
    
    #identify red areas
    red = (r - g > 50) & (r - b > 60)
    red = red.astype(np.uint8)
    kernel = np.ones((2,2), np.uint8)
    red = cv2.erode(red, kernel)
    kernel = np.ones((7,7), np.uint8)
    red = cv2.dilate(red, kernel)
    red = cv2.bitwise_and(field, red)
    red_players = red
    
    #identify blue areas
    blue = (b - g > 252) & (b  - r > 252)
    blue = blue.astype(np.uint8)
    blue = cv2.bitwise_and(field, blue)
    kernel = np.ones((1,1), np.uint8)
    blue = cv2.erode(blue, kernel)
    kernel = np.ones((7,7), np.uint8)
    blue = cv2.dilate(blue, kernel)
    blue_players = blue
    
    #Get centroids of red players
    nb_components, output, stats, red_player_centroids = cv2.connectedComponentsWithStats(red, connectivity=4)
    
    #Get centroids of blue players
    nb_components, output, stats, blue_player_centroids = cv2.connectedComponentsWithStats(red, connectivity=4)
        

    
    

    return red_player_centroids, blue_player_centroids, field, red_players, blue_players
