import numpy as np
import cv2
import pathlib.Path
import warnings

"""
TODO:
 - Implement ImageSource
"""

# Lenghts are mesured in mm

def save_flatten_param(ref_images_path: str, saved_param_path: str):
    chessboard_shape = (9, 6)
    chessboeard_square_real_dim = 30

    ref_images_floder = pathlib.Path(ref_images_path)
    image_res = None
    for im_file in ref_images_floder.iterdir():
        # load ref image and converts it do grayscales
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        if not image_res is None:
            image_res = np.shape(im)

        #search for the chessboard in it
        success, corners = cv2.findChessboardCorners(im, chessboard_shape)

        if not success:
            warnings.warn(f"failed do find chessboard in {im_file.name}")

    # Save param
    np.save(saved_param_path + "/cameraMatrix", cameraMatrix)
    np.save(saved_param_path + "/distCoefs", distCoefs)

def load_flatten_param(saved_param_path) -> tuple:
    cameraMatrix = np.load(saved_param_path + "/cameraMatrix.npy")
    distCoefs = np.load(saved_param_path + "/distCoefs.npy")
    return cameraMatrix, distCoefs

aruco_detector = None

tags_found_in_source = {}
def find_arucos(source: ImageSource):
    global tags_found_in_source, aruco_detector
    im = source.get_next_img()

    # detect tags
    found_corners, ids, _ = aruco_detector.detectMarkers(im)

    # format found_corners
    new_corners_shape = (len(found_corners), 4, 2)
    found_corners = np.reshape(found_corners, new_corners_shape)

    # format into a dict of list of tags found indexed by id
    tags_found_in_source = {}
    for corners, id in zip(found_corners, ids):
        if id in tags_found_in_source:
            tags_found_in_source[id].append(corners)
        else:
            tags_found_in_source[id] = [corners]
