'''
This tool is used for calibrating a camera using a video of a chessboard. It generates a calibration configuration that can be used by other tools in the project.
To use this tool, provide the path to the video file as the input argument and the path to the output file where the calibration configuration will be saved.
The calibration configuration includes the intrinsic matrix, distortion coefficients, camera matrix, rotation vectors, translation vectors, and region of interest (ROI).
The generated calibration configuration can be used by other tools in the project for tasks such as depth prediction, point cloud generation, and testing on different cameras.

Note: This tool requires a physical print of an OpenCV chessboard pattern. More specifically, we need to calibrate off of a 9x6 checkerboard grid with each square being 30mm x 30mm.
'''


from typing import Generator, Optional, Tuple, List, Self, Any
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tqdm import tqdm

import multiprocessing as mp
import numpy as np
import cv2 as cv
import argparse
import asyncio
import random
import json


CRITERIA = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001
CALIBRATION = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK


@dataclass(frozen=True)
class Calibration:
    intrinsic: np.ndarray
    distortion: np.ndarray
    matrix: np.ndarray
    rvecs: np.ndarray
    tvecs: np.ndarray
    roi: Tuple[int, int, int, int]

    @staticmethod
    def calibrate(objpoints: List[np.ndarray], imgpoints: List[np.ndarray], shape: Tuple[int, int]) -> Optional[Self]:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints,
            imgpoints,
            shape,
            None,
            None,
        )

        if not ret:
            return None

        intrinsic, roi = cv.getOptimalNewCameraMatrix(
            mtx,
            dist,
            shape,
            1,
            (0, 0),
        )

        return Calibration(intrinsic, dist, mtx, rvecs, tvecs, roi)


    @staticmethod
    def find_corners(objp: np.ndarray, grid_dims: Tuple[int, int], frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(frame, grid_dims, CALIBRATION)

        if not ret:
            return None

        return objp, cv.cornerSubPix(frame, corners, (11, 11), (-1, -1), CRITERIA)


    @staticmethod
    def load(path: Path) -> Self:
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
    

    def error(self, objpoints: List[np.ndarray], imgpoints: List[np.ndarray]) -> float:
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
        return cv.undistort(frame, self.matrix, self.distortion, None, self.intrinsic)


    def rectify(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w]
    

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
    

@contextmanager
def capture(*args, **kwargs) -> Generator[cv.VideoCapture, None, None]:
    cap = cv.VideoCapture(*args, **kwargs)
    try:
        yield cap
    
    finally:
        cap.release()


def capture_triple(cap: cv.VideoCapture) -> Tuple[int, int, int]:
    count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                 
    return count, width, height


def stream(cap: cv.VideoCapture) -> Generator[np.ndarray, None, None]:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame is not None:
            yield frame


def gen_objp(grid_dims: Tuple[int, int], square_width: int) -> List[np.ndarray]:
    objp = np.zeros((np.prod(grid_dims), 3), np.float32)
    objp[:, :2] = np.indices(grid_dims).T.reshape(-1, 2) * square_width

    return objp


def calibrate(args: object) -> Optional[Calibration]:
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor, capture(str(args.input)) as cap:
        count, width, height = capture_triple(cap)

        objp = gen_objp(args.grid_dims, args.square_width)
        applicative = partial(Calibration.find_corners, objp, args.grid_dims)

        corners = tqdm(executor.map(applicative, stream(cap)), total=count, desc='Finding corners')
        objpoints, imgpoints = zip(*[point for point in corners if point is not None])

    objsample = random.sample(objpoints, args.sample_amount)
    imgsample = random.sample(imgpoints, args.sample_amount)

    print('[STATUS]: Calibrating...')
    if calibration := Calibration.calibrate(objsample, imgsample, (width, height)):
        print(f'Error: {calibration.error(objpoints, imgpoints)}')
        calibration.dump(args.output)
        
    else:
        print('[ERROR]: No chessboard corners found in the sampled frames\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate a camera using a video of a chessboard')
    parser.add_argument('input', type=Path, help='Path to the video file')
    parser.add_argument('output', type=Path, help='Path to the output file')
    parser.add_argument('--grid-dims', type=int, nargs=2, default=(9, 6), help='Grid dimensions (width height)')
    parser.add_argument('--square-width', type=float, default=30, help='Width of each square')
    parser.add_argument('--sample-amount', type=int, default=30, help='Number of calibration frames to sample')

    calibrate(parser.parse_args())