from dev_env_vars import *
import cv2
KNOWN_DISTANCE = 24.0 #
KNOWN_WIDTH = 11.0
def know_distance(widthPx):
    # initialize the known distance from the camera to the object, which
    # in this case is 24 inches
    #KNOWN_DISTANCE = 24.0
    # initialize the known object width, which in this case, the piece of
    # paper is 12 inches wide
    # KNOWN_WIDTH = 11.0
    # load the furst image that contains an object that is KNOWN TO BE 2 feet
    # from our camera, then find the paper marker in the image, and initialize
    # the focal length
    focalLength = (widthPx * KNOWN_DISTANCE) / KNOWN_WIDTH
    return focalLength

def distance_to_camera(KNOWN_WIDTH, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (KNOWN_WIDTH * focalLength) / perWidth


def draw_distance(detections,image):
    dist_t_camera_p = 0
    dist_t_camera_c = 0
    person_no = 0
    for cat, score, bounds in detections:
        # print(cat)
        if "person" == bytes.decode(cat):
            person_no += 1
            dist_t_camera_p = distance_to_camera(7, know_distance(Xresolution ), bounds[2])
            if dist_t_camera_p < min_distance:
                print("stop and go back")
        if "cell phone" == bytes.decode(cat):
            dist_t_camera_c = distance_to_camera(5, know_distance(Xresolution), bounds[2])
    cv2.putText(image, "DIstance PERSON: " + str(int(dist_t_camera_p)), (20, 60), cv2.FONT_HERSHEY_COMPLEX, font_size,
                azzure)

    cv2.putText(image, "DIstance Cell: " + str(int(dist_t_camera_c)), (20, 80), cv2.FONT_HERSHEY_COMPLEX, font_size,
                azzure)
    cv2.putText(image, "NO of persons: " + str(int(person_no)), (20, 40), cv2.FONT_HERSHEY_COMPLEX, font_size,
                azzure)
    return image,

