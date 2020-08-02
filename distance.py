
KNOWN_DISTANCE = 24.0
KNOWN_WIDTH = 11.0
def know_distance(widthPx):
    # initialize the known distance from the camera to the object, which
    # in this case is 24 inches
    # KNOWN_DISTANCE = 24.0
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


