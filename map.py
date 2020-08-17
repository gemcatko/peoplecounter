import cv2
import numpy as np
from dev_env_vars import *

def map_visualization(objects):
    zoom = 2        # how many times smorel
    map_X = int(Xresolution/zoom)
    map_Y = int(Xresolution/zoom)
    middle = int(map_X/2)

    map_frame= np.zeros((map_X, map_Y, 3), np.uint8)

    for id in objects:

        x = int(objects[id].bounds[0]/zoom)           #x
        y =  int((objects[id].distance )/zoom)+middle        #dis
        #radius = int(objects[id].bounds[3]/zoom)       #w
        radius = 5
        print("X,y",x,y,objects[id].id)
        cv2.circle(map_frame, (x, y), radius, (0, 0, 255), 5)
        #cv2.circle(map_frame, (250, 250), 5, red, 5)

    cv2.circle(map_frame, (middle, middle), 5, (0, 255, 255), 5)
    cv2.imshow('map1', map_frame)
    #image = cv2.circle(image, center_coordinates, radius, color, thickness)
    return map_frame



