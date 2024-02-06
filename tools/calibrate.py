#!/usr/bin/env python

import numpy as np
import cv2 as cv
import argparse
import json
import sys
import random
def calibrate(video_path: str, show: bool, num_frames: int) -> None:
    #termination criteria
    CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from sampled frames.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv.VideoCapture(video_path)

    # Get total number of frames in the video
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Sample random frames
    sampled_frames = random.sample(range(total_frames), num_frames)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count not in sampled_frames:
            frame_count += 1
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,6), None)

        if ret is False:
            sys.stderr.write('Chessboard corners not found in frame\n')
            continue
        
        # If found, add object points, image points (after refining them)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), CRITERIA)
        imgpoints.append(corners2)

        if show:
            cv.drawChessboardCorners(frame, (7,6), corners2, ret)
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    cv.destroyAllWindows()

    if len(objpoints) == 0:
        sys.stderr.write('No chessboard corners found in the sampled frames\n')
        sys.exit(1)

    # Calibrate the camera
    _, mtx, dist, _, _ = cv.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None,
    )

    # Get the optimal new camera matrix
    intrinsic_matrix, _ = cv.getOptimalNewCameraMatrix(
        mtx,
        dist,
        (frame.shape[1], frame.shape[0]),  # Use the frame width and height
        1,
        (0, 0),
    )

    sys.stdout.write(f'{intrinsic_matrix}\n')
    #sys.stdout.write(f'{json.dump(intrinsic_matrix)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument(
        'video',
        metavar='video',
        type=str,
        help='input video path',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='display the results of the calibration',
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=10,
        help='number of frames to sample from the video',
    )
    args = parser.parse_args()

    calibrate(args.video, args.show, args.num_frames)
