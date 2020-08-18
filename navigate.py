import logging
from dev_env_vars import *
from datetime import datetime

def navigate(distance_results):
    for distance,id,cat, score, bounds in distance_results:
        try:
            if "person" == bytes.decode(cat):
                if not to_close(distance):
                    cur_time = str(datetime.now().time()) + " GO"
                    detection = [distance,cat, score, bounds]
                    rotate_to_target(detection)
                    #logging.info(cur_time)

            if "cell phone" == bytes.decode(cat):

                if not to_close(distance):
                    cur_time = str(datetime.now().time()) + " GO"
                    #logging.info(cur_time)

        except Exception as e:
            print(e)

def to_close(distance):
    if distance < min_distance:
        cur_time = str(datetime.now().time()) + " BACKWARD"
        #logging.info(cur_time)
        return True

def rotate_to_target (detection):

    x, y, w, h = detection[3][0], \
                         detection[3][1], \
                         detection[3][2], \
                         detection[3][3]
    middleX= Xresolution / 2
    middleY= Yresolution / 2
    darknetvscameraresolutionx = (Xresolution / network_width)
    darknetvscameraresolutiony = (Yresolution / network_heigth)
    x = x * darknetvscameraresolutionx
    y = y * darknetvscameraresolutiony
    w = w * darknetvscameraresolutionx
    h = h * darknetvscameraresolutiony
    try:
        if x < middleX:
             #logging.info("Rotate to left")
            pass
        if x > middleX:
           # logging.info("Rotate to right")
            pass
    except Exception as e:
        print(e)

