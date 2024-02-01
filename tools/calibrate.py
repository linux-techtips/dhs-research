#!/usr/bin/env python

import numpy as np
import cv2 as cv
import argparse
import sys


def calibrate(fname: str, show: bool) -> None:
    #termination criteria
    CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    if ret is False:
        sys.stderr.write(f'Chessboard corners not found in {fname}\n')
        sys.exit(1)
    
    # If found, add object points, image points (after refining them)
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), CRITERIA)
    imgpoints.append(corners2)

    # Calibrate the camera
    _ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None,
    )

    # Get the optimal new camera matrix
    intrinsic_matrix, roi = cv.getOptimalNewCameraMatrix(
        mtx,
        dist,
        (img.shape[1], img.shape[0]),  # Use the image width and height
        1,
        (0, 0),
    )

    sys.stdout.write(f'{intrinsic_matrix}\n')

    if show:
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey()
        cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument(
        'images',
        metavar='image',
        type=str,
        nargs='+',
        help='input image paths',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='display the results of the calibration',
    )
    args = parser.parse_args()

    calibrate(args.images[0], args.show)