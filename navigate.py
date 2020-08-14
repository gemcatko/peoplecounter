import logging
from dev_env_vars import *
from datetime import datetime

def navigate(distance_results):
    for distance,cat, score, bounds in distance_results:
        try:
            if "person" == bytes.decode(cat):
                if not to_close(distance):
                    cur_time = str(datetime.now().time()) + " GO"
                    logging.info(cur_time)

            if "cell phone" == bytes.decode(cat):

                if not to_close(distance):
                    cur_time = str(datetime.now().time()) + " GO"
                    logging.info(cur_time)

        except Exception as e:
            print(e)

def to_close(distance):
    if distance < min_distance:
        cur_time = str(datetime.now().time()) + " BACKWARD"
        logging.info(cur_time)
        return True

