import os
import cv2
import time
import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/home/automateit/Projects/darknet-alexeyAB/darknet')
#sys.path.insert(1, '/home/ce/Projects/darknet_alex_darknet')
sys.path.insert(1, '/home/solcanma/darknet')
import darknet
from multiprocessing import Process, Value, Array, Manager
from multiprocessing import shared_memory
import numpy as np
from dev_env_vars import *
manager = Manager()
manager_detections = manager.list()
from pyimagesearch.centroidtracker import CentroidTracker
import logging

shm = shared_memory.SharedMemory(create=True, size=6520800, name='psm_c013ddb6')
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
                    green, 2)
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


    #cap = cv2.VideoCapture(video_filename_path)
    cap.set(3, Xresolution)
    cap.set(4, Yresolution)

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
        frame_rgb = rotate_by_angel_and_delay(frame_rgb,0,delay_off_whole_program)  # use for changing direction of video and speed of video
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

class YObject:
    # use for creating objects from Yolo.
    # def __init__(self, centroid_id, category, score, bounds):
    def __init__(self, id, category, score, bounds):
        # copy paste functionality of  detect_object_4_c
        self.id = id
        self.category = category
        self.score = score
        self.bounds = bounds
        self.is_big = False
        self.is_detected_by_detector = True
        self.ignore = False
        self.is_picture_saved = False
        # self.ready_for_blink_start = False
        # self.ready_for_blink_end = False
        global frame

    def draw_object_bb(self, frame, idresults):
        """
        Draw objects name to CV2 frame  using cv2 only if  detected by detector
        :return: none
        """
        if self.is_detected_by_detector or id in (item for sublist in idresults for item in sublist):
            x, y, w, h = self.bounds
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), green, 1)

    def draw_category(self, frame, idresults):
        """
        Draw objects classto CV2 frame  using cv2 only if  detected by detector
        :return: none
        """
        if self.is_detected_by_detector or id in (item for sublist in idresults for item in sublist):
            x, y, w, h = self.bounds
            cv2.putText(frame, str(self.category), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 0))

    def draw_object_score(self, frame, idresults):
        """
        Draw objects score to CV2 frame  using cv2 only if still detected by detector
        :return: none
        """
        if self.is_detected_by_detector or id in (item for sublist in idresults for item in sublist):
            x, y, w, h = self.bounds
            cv2.putText(frame, str(round(self.score, 2)), (int(x - 20), int(y - 20)), cv2.FONT_HERSHEY_COMPLEX, font_size,
                        azzure)

    def draw_object_id(self, frame, idresults):
        """
        Drae object Id to CV2 frame only if still detected by detector
        :return:
        """
        if self.is_detected_by_detector or id in (item for sublist in idresults for item in sublist):
            x, y, w, h = self.bounds
            cv2.putText(frame, "ID" +str(self.id), (int(x - 30), int(y)), cv2.FONT_HERSHEY_COMPLEX, font_size, azzure)

    def draw_object_position_on_trail(self, frame, idresults):
        """
        Drae object Id to CV2 frame only if still detected by detector
        :return:
        """
        if self.is_detected_by_detector or id in (item for sublist in idresults for item in sublist):
            x, y, w, h = self.bounds
            x_rel, y_rel, w_rel, h_rel, area_rel = calculate_relative_coordinates(x, y, w, h)
            # position_on_trail_for_screen = round(self.position_on_trail + (x_rel * size_of_one_screen_in_dpi), 1)
            position_on_trail_for_screen = round((x_rel * size_of_one_screen_in_dpi))
            cv2.putText(frame, str(position_on_trail_for_screen), (int(x), int(y + 25)), cv2.FONT_HERSHEY_COMPLEX, font_size,
                        (yellow))

    def save_picure_of_every_detected_object(self, file_name="detected_objects"):
        """
        :param folder name need to be suplied:
        """
        if self.is_picture_saved == False:
            save_picture_to_file(file_name)
        self.is_picture_saved = True

def update_resutls_for_id(results, ct_objects):
    """
    loop over the tracked objects from Yolo34
    Reconstruct Yolo34 results with object id (data from centroid tracker) an put object ID to idresults list, like :
    class 'list'>[(b'person', 0.9972826838493347, (646.4600219726562, 442.1628112792969, 1113.6322021484375, 609.4992065429688)), (b'bottle', 0.5920438170433044, (315.3851318359375, 251.22744750976562, 298.9032287597656, 215.8708953857422))]
    class 'list'>[(1, b'person', 0.9972826838493347, (646.4600219726562, 442.1628112792969, 1113.6322021484375, 609.4992065429688)), (4, b'bottle', 0.5920438170433044, (315.3851318359375, 251.22744750976562, 298.9032287597656, 215.8708953857422))]
    :param results from Darknet, ct_objects:
    :return:idresults
    """
    idresults = []
    try:
        for cat, score, bounds in results:
            x, y, w, h = bounds
            # loop over the tracked objects from Centroid
            for (objectID, centroid) in ct_objects.items():
                # put centroid coordinates to cX and Cy variables
                cX, cY = centroid[0], centroid[1]
                # there is difference between yolo34 centroids and centroids calculated by centroid tracker,Centroid closer then 2 pixel are considired to matcg  TODO find where?
                if abs(cX - int(x)) <= 2 and abs(cY - int(y)) <= 2:
                    # reconstruct detection list as from yolo34 including ID from centroid
                    idresult = objectID, cat, score, bounds
                    idresults.append(idresult)
        return idresults
    except:
        return idresults


def show_fps(start_time, end_time, name_of_frame):
    duration_of_loop = end_time - start_time
    FPS = round(1 / duration_of_loop, 1)
    cv2.putText(name_of_frame, str(FPS), (int(Xres - 80), int(Yres - 40)), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 100, 255))
    # print(FPS)
    return FPS


def is_Yobject_to_big(bounds):
    x, y, w, h = bounds
    # cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), blue, 4)
    if int(x + w / 2) > (Xres - Xres / 4):
        if w > (Xres - Xres / 2):
            print("objekt is too big")
            return True
    return False


def second_visualization(net_width, net_heigth):
    existing_shm = shared_memory.SharedMemory(name='psm_c013ddb6')
    image = np.ndarray((net_width, net_heigth, 3), dtype=np.uint8, buffer=existing_shm.buf)
    ct = CentroidTracker(maxDisappeared=2000)
    which_id_to_delete = 0  # is used for object deletion start
    start_time = time.time()  # for FPS counting
    while True:
        frame = image
        # get data from shared memory
        results = manager_detections
        results = unpack_results(results)
        # print("Results from darkent:",results)
        # genreate IDs for for results from Darknet
        ct_objects = ct.update(convert_bounding_boxes_form_Yolo_Centroid_format(results))
        idresults = update_resutls_for_id(results, ct_objects)
        # print("Resultswith ID", idresults)
        cv2.waitKey(3)
        for id, category, score, bounds in idresults:
            try:  #
                # if Yobjekt with specific id already exists, update it
                # TODO # to je mozno chyba,co sa stane z objektami ktorych id je este na zobrazene ale nuz je objekt dissapeared
                if objekty[id].id == id:  # and objekty[id].category == category.decode("utf-8"):
                    # if objekty[id].category == category.decode("utf-8"): #- If you enable you need to take care of umached objects which are deteceted
                    if not objekty[id].is_big:
                        objekty[id].category = category.decode("utf-8")
                        objekty[id].score = score
                        objekty[id].bounds = bounds

                        objekty[id].is_detected_by_detector = True
                        if is_Yobject_to_big(bounds):           # if objects is across whole camera view it need to go to trail visulaization so it can be marked. If not done centroid tracker whould hold it on  visible screen and it would be marked as very smal object when leaving camera view
                            objekty[id].is_big = True

            except:
                # create new object if not existing
                objekty[id] = YObject(id, category.decode("utf-8"), score, bounds)
        if len(objekty) > max_Yobject:  # max number of object which to keep
            del objekty[which_id_to_delete]
            which_id_to_delete = which_id_to_delete + 1
        if count_visible_objekt("person",objekty) > 0:
            cv2.putText(frame, "Number of person on screen:" + str(count_visible_objekt("person",objekty)), (20, 20), cv2.FONT_HERSHEY_COMPLEX,font_size,azzure)
            print(count_visible_objekt("person",objekty))
        for id in objekty:
            objekty[id].draw_object_bb(frame, idresults)
            #objekty[id].draw_category(frame, idresults)
            #objekty[id].draw_object_score(frame, idresults)
            objekty[id].draw_object_id(frame, idresults)
            #objekty[id].draw_object_position_on_trail(frame, idresults)
            # objekty[id].save_picure_of_every_detected_object("detected_objects")
        update_objekty_if_not_detected(objekty, idresults)
        """
        try:
            check_on_vysialization(draw_trail_visualization(objekty, s_distance), objekty, s_distance)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        """
        end_time = time.time()
        #show_fps(start_time, end_time, frame)
        start_time = time.time()
        cv2.imshow('second_visualization', frame)
        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break

def count_visible_objekt(object_to_count, objekty):
    count_of_objects = 0
    for id in objekty:
        if objekty[id].is_detected_by_detector == True:
            count_of_objects += 1
    return count_of_objects



def update_objekty_if_not_detected(objekty, idresults):
    """
    :param objekty:
    Is updating all objects store in objekty if not in in idresults list from detector
    :return:
    """
    for id in objekty:
        # id in (item for sublist in idresults for item in sublist) is returning True or False good explanation is https://www.geeksforgeeks.org/python-check-if-element-exists-in-list-of-lists/
        if not id in (item for sublist in idresults for item in sublist):
            objekty[id].is_detected_by_detector = False



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




second_visualization_proc = Process(target=second_visualization, args=(network_width, network_heigth))
second_visualization_proc.daemon = True

if __name__ == "__main__":
    second_visualization_proc.start()
    YOLO()
