'''
This tool is used for calibrating a camera using a video of a chessboard. It generates a calibration configuration that can be used by other tools in the project.
To use this tool, provide the path to the video file as the input argument and the path to the output file where the calibration configuration will be saved.
The calibration configuration includes the intrinsic matrix, distortion coefficients, camera matrix, rotation vectors, translation vectors, and region of interest (ROI).
The generated calibration configuration can be used by other tools in the project for tasks such as depth prediction, point cloud generation, and testing on different cameras.

Note: This tool requires a physical print of an OpenCV chessboard pattern. More specifically, we need to calibrate off of a 9x6 checkerboard grid with each square being 30mm x 30mm.
'''

<<<<<<< HEAD

from typing import Generator, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor
=======
from multiprocessing import ProcessPoolExecutor
from typing import Optional, Generator
>>>>>>> cddbfcb429f2812dda420cd9b0e44548be8c122d
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tqdm import tqdm # type : ignore[untyped-import]

import multiprocessing as mp
import numpy as np
import cv2 as cv
import argparse
<<<<<<< HEAD
import random
=======
>>>>>>> cddbfcb429f2812dda420cd9b0e44548be8c122d
import json


CRITERIA = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001
CALIBRATION = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK


@dataclass(frozen=True)
class Calibration:
    distortion: np.ndarray
    intrin: np.ndarray
    matrix: np.ndarray
    rvecs: np.ndarray
    tvecs: np.ndarray
    roi: tuple[int, int, int, int]


    @staticmethod
<<<<<<< HEAD
    def calibrate(objpoints: List[np.ndarray], imgpoints: List[np.ndarray], shape: Tuple[int, int]) -> Optional['Calibration']:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints,
            imgpoints,
            shape,
            None,
            None,
        )
=======
    def calibrate(objpoints: list[np.ndarray], imgpoints: list[np.ndarray], shape: tuple[int, int]) -> Optional['Calibration']:
        ret, matrix, distortion, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, shape, None, None)
>>>>>>> cddbfcb429f2812dda420cd9b0e44548be8c122d

        if not ret:
            return None

        intrin, roi = cv.getOptimalNewCameraMatrix(
            matrix, distortion, shape, 1, (0, 0))

        return Calibration(distortion, intrin, matrix, rvecs, tvecs, roi)

    
    @staticmethod
    def find_corners(objpoints: np.ndarray, grid_dims: tuple[int, int], frame: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(frame, grid_dims, CALIBRATION)

        if not ret:
            return None

        return objpoints, cv.cornerSubPix(frame, corners, (11, 11), (-1, -1), CRITERIA)


    @staticmethod
    def load(path: Path) -> 'Calibration':
        with path.open('r') as file:
            data = json.load(file)

            return Calibration(
                np.array(data['intrinsic_matrix']),
                np.array(data['distortion_coefficients']),
                np.array(data['matrix']),
                [np.array(rvec) for rvec in data['rvecs']],
                [np.array(tvec) for tvec in data['tvecs']],
                tuple(data['roi']),
            )


    def dump(self, path: Path) -> None:
        with path.open('w') as file:
            json.dump({
                "intrinsic_matrix": self.intrinsic.tolist(),
                "distortion_coefficients": self.distortion.tolist(),
                "matrix": self.matrix.tolist(),
                "rvecs": [rvec.tolist() for rvec in self.rvecs],
                "tvecs": [tvec.tolist() for tvec in self.tvecs],
                "roi": self.roi,
        }, file, indent=2)


    def error(self, objpoints: list[np.ndarray], imgpoints: list[np.ndarray]) -> float:
        '''
        Calculates the mean reprojection error of the calibration.
        Note: If a calibration is particularly bad, OpenCV might fail an assertion.
        '''

        mean = 0.0
        for objpoint, imgpoint, rvec, tvec in zip(objpoints, imgpoints, self.rvecs, self.tvecs):
            points, _ = cv.projectPoints(objpoint, rvec, tvec, self.intrinsic, self.distortion)
            mean += cv.norm(imgpoint, points, cv.NORM_L2) / len(points)

        return mean / len(objpoints)
    

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        '''
        Undistorts a given frame based off of the calibration's distortion coefficent.
        '''
        
        return cv.undistort(frame, self.matrix, self.distortion, None, self.intrinsic)


    def rectify(self, frame: np.ndarray) -> np.ndarray:
        '''
        Crops the image to the region of interest (ROI) of the calibration. This will be useful for depth prediction.
        '''
        
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w]


@contextmanager
def Capture(*args, **kwargs) -> Generator[cv.VideoCapture, None, None]:
<<<<<<< HEAD
    # why is this not built into OpenCV??
    cap = cv.VideoCapture(*args, **kwargs)
=======
    cap = cv.VideoCapture(*args , **kwargs)
    
>>>>>>> cddbfcb429f2812dda420cd9b0e44548be8c122d
    try:
        yield cap

    finally:
        cap.release()


def stream(cap: cv.VideoCapture) -> Generator[np.ndarray, None, None]:
    # why is this also not built into OpenCV???
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame is not None:
            yield frame


<<<<<<< HEAD
def gen_objp(grid_dims: Tuple[int, int], square_width: int) -> List[np.ndarray]:
    '''
    Generates the list of points for a physical chessboard pattern.
    Since we know the grid dimensions and the size of each square, we can generate mappings in which 
    to relate the physical imgpoints to.
    '''

=======
def capture_triple(cap: cv.VideoCapture) -> tuple[int, int, int]:
    count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                 
    return count, width, height


def gen_objp(grid_dims: tuple[int, int], square_width: int) -> list[np.ndarray]:
>>>>>>> cddbfcb429f2812dda420cd9b0e44548be8c122d
    objp = np.zeros((np.prod(grid_dims), 3), np.float32)
    objp[:, :2] = np.indices(grid_dims).T.reshape(-1, 2) * square_width

    return objp


<<<<<<< HEAD
def show_results(calibration: Calibration, frame: np.ndarray) -> None:
    undistorted = calibration.undistort(frame)
    rectified = calibration.rectify(undistorted)

    cv.imshow('Original', frame)
    cv.imshow('Undistorted', undistorted)
    cv.imshow('Rectified', rectified)

    cv.waitKey(0)
    cv.destroyAllWindows()


def calibrate(args: object) -> None:
=======
def calibrate(args: object) -> Optional[Calibration]:
>>>>>>> cddbfcb429f2812dda420cd9b0e44548be8c122d
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor, Capture(str(args.input)) as cap:
        count, width, height = capture_triple(cap)
        frames = list(stream(cap))

        objp = gen_objp(args.grid_dims, args.square_width)
        applicative = partial(Calibration.find_corners, objp, args.grid_dims) # performing partial application in case generating objpoints becomes more novel

        corners = tqdm(executor.map(applicative, frames), total=count, desc='Finding corners')
        objpoints, imgpoints = zip(*[point for point in corners if point is not None])

<<<<<<< HEAD
    # TODO (Carter Vavra): Implement a better sampling method. This is a bit naive.

    objsample = random.sample(objpoints, args.sample_amount)
    imgsample = random.sample(imgpoints, args.sample_amount)
=======
    objsample = np.random.sample(objpoints, args.sample_amount)
    imgsample = np.random.sample(imgpoints, args.sample_amount)
>>>>>>> cddbfcb429f2812dda420cd9b0e44548be8c122d

    print('[STATUS]: Calibrating...')
    if calibration := Calibration.calibrate(objsample, imgsample, (width, height)):
        print(f'Error: {calibration.error(objpoints, imgpoints)}')
        calibration.dump(args.output)

        if args.show_results:
            show_results(calibration, random.choice(frames))

    else:
        print('[ERROR]: No chessboard corners found in the sampled frames\n')
    

if __name__ == '__main__':
    # TODO (Carter Vavra): Allow user to specify Calibration Criteria

    parser = argparse.ArgumentParser(description='Calibrate a camera using a video of a chessboard')
    parser.add_argument('input', type=Path, help='Path to the video file')
    parser.add_argument('output', type=Path, help='Path to the output file')
    parser.add_argument('--grid-dims', type=int, nargs=2, default=(9, 6), help='Grid dimensions (width height)')
    parser.add_argument('--square-width', type=float, default=30, help='Width of each square in mm')
    parser.add_argument('--sample-amount', type=int, default=30, help='Number of calibration frames to sample')
    parser.add_argument('--show-results', action='store_true', default=False, help='Show the results of the calibration')

    calibrate(parser.parse_args())