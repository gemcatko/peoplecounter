import os
import cv2
import time
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/automateit/Projects/darknet-alexeyAB/darknet')
import darknet
from multiprocessing import Process, Value, Array, Manager
from multiprocessing import shared_memory
import numpy as np
from dev_env_vars import *
manager = Manager()
manager_detections = manager.list()
from pyimagesearch.centroidtracker import CentroidTracker
import logging

shm = shared_memory.SharedMemory(create=True, size=6520800, name='psm_c013ddb9')
shm_image = np.ndarray((network_width, network_heigth, 3), dtype=np.uint8, buffer=shm.buf)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] (%(threadName)-10s) %(message)s', )


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
    start_time = time.time()

    global metaMain, netMain, altNames, manager_detections
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    # cap = cv2.VideoCapture(0)
    # USE if  webcam
    """
    cap = cv2.VideoCapture(0)  # set web cam properties width and height, working for USB for webcam
    cap.set(3, Xresolution)
    cap.set(4, Yresolution)
    ##Use webcam with high frame rate
    codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    cap.set(cv2.CAP_PROP_FPS, 50)  # FPS60FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Xresolution)  # set resolutionx of webcam
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Yresolution)  # set resolutiony of webcam
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    print(cap.get(cv2.CAP_PROP_FPS))
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    """

    cap = cv2.VideoCapture(video_filename_path)
    #cap.set(3, Xresolution)
    #cap.set(4, Yresolution)

    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_rgb = rotate_by_angel_and_delay(frame_rgb,270,delay_off_whole_program)  # use for changing direction of video and speed of video
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        del manager_detections[:]  # need to be cleared every iterration
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=detection_treshold)
        # print(detections)
        manager_detections.append(detections)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shm_image[:] = image[:]  # copy image to shared memory as array because we would like to share with other proces

        end_time = time.time()
        show_fps(start_time, end_time, image)
        start_time = time.time()
        cv2.imshow('Yolo_out', image)
        cv2.waitKey(3)
    cap.release()
    out.release()


def convert_bounding_boxes_form_Yolo_Centroid_format(results):
    # clean rect so it is clean an can be filled with new detection from frame\
    # later used in conversion_to_x1y1x2y2 . Conversion from yolo format to Centroid Format
    # rects are needed for centroid to work. They need to be cleared every time
    rects = []

    # if len(results) <= 1: # check if array is not empty, for prevention of crashing in later stage
    #    return []
    try:
        for cat, score, bounds in results:  # unpacking
            x, y, w, h = bounds
            """
            convert from yolo format to cetroid format
            Yolo output:
            [(b'person', 0.9299755096435547, (363.68475341796875, 348.0577087402344, 252.04286193847656, 231.17266845703125)), (b'vase', 0.3197628855705261, (120.3013687133789, 405.3641357421875, 40.76551055908203, 32.07142639160156))]
            [(b'mark', 0.9893345236778259, (86.11815643310547, 231.90643310546875, 22.100597381591797, 54.182857513427734)), (b'mark', 0.8441593050956726, (225.28382873535156, 234.5716094970703, 14.333066940307617, 53.428749084472656)), (b'edge', 0.6000953316688538, (377.6446838378906, 254.71759033203125, 8.562969207763672, 18.379894256591797)), (b'edge', 0.5561915636062622, (388.4414367675781, 211.0662841796875, 10.678437232971191, 15.206807136535645)), (b'edge', 0.44139474630355835, (377.0844421386719, 150.8873748779297, 9.128596305847168, 18.9124755859375)), (b'crack', 0.28897273540496826, (268.6462707519531, 169.00457763671875, 253.9573516845703, 34.764007568359375))]]
            """
            # calculate bounding box for every object from YOLO for centroid purposes
            box = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            # append to list of  bounding boxes for centroid
            rects.append(box.astype("int"))
        return rects
    except:
        # print("There was a problem with extrection from result:", rects)
        return rects


def unpack_results(results):
    for oneresult in results:  # unpacking
        # print(oneresult)
        return oneresult


def calculate_relative_coordinates(x, y, w, h):
    """
    Calculate coordinates in percentage relative to the screen
    :param x: center of detected object on x axis in pixels
    :param y: center of detected object on y axis in pixels
    :param w: width of detected object on x axis in pixels
    :param h: height of detected object on y axis in pixels
    :return: x_rel, y_rel, w_rel, h_rel, area_rel
    """
    x_rel = x / Xres
    y_rel = y / Yres
    w_rel = w / Xres
    h_rel = h / Yres
    area_rel = w_rel * h_rel
    return x_rel, y_rel, w_rel, h_rel, area_rel


def rotate_by_angel_and_delay(frame,angel, delay):
    (h, w) = frame.shape[:2]
    center = (w / 2, h / 2)
    angle180 = angel
    scale = 1.0
    # 180 degrees
    M = cv2.getRotationMatrix2D(center, angle180, scale)
    rotated = cv2.warpAffine(frame, M, (w, h))
    time.sleep(delay)
    return rotated

def show_fps(start_time, end_time, name_of_frame):
    duration_of_loop = end_time - start_time
    FPS = round(1 / duration_of_loop, 1)
    cv2.putText(name_of_frame, str(FPS), (int(Xres - 80), int(Yres - 40)), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 100, 255))
    # print(FPS)
    return FPS
#second_visualization_proc = Process(target=second_visualization, args=(network_width, network_heigth))
#second_visualization_proc.daemon = True

if __name__ == "__main__":
    #second_visualization_proc.start()
    YOLO()