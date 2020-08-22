import logging
from dev_env_vars import *
from datetime import datetime

def navigate_o(objects):
    """

    :param objects:
        self.id = id
        self.category = category
        self.score = score
        self.bounds = bounds
        self.is_big = False
        self.is_detected_by_detector = True
        self.ignore = False
        self.is_picture_saved = False
        self.distance = distance
    :return:
    """
    for id in objects:
        try:
            if "person" == objects[id].category:
                if not to_close(objects[id].distance):
                    cur_time = str(datetime.now().time()) + " GO"
                    detection = [objects[id].distance,objects[id].category,objects[id].score,objects[id].bounds]
                    rotate_to_target(detection)
                    logging.info(cur_time)

            if "cell phone" == objects[id].category:

                if not to_close(objects[id].distance):
                    cur_time = str(datetime.now().time()) + " GO"
                    logging.info(cur_time)

        except Exception as e:
            print(e)


def to_close(distance):
    if distance < min_distance:
        cur_time = str(datetime.now().time()) + " BACKWARD"
        logging.info(cur_time)
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

