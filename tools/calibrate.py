from typing import TypeAlias, Optional, Tuple, List, Self
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import multiprocessing as mp
import numpy as np
import cv2 as cv
import argparse
import json
import sys


CornerResult: TypeAlias = Optional[Tuple[np.ndarray, np.ndarray]]


CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
OBJP = np.zeros((6*7, 3), np.float32)
OBJP[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)


@dataclass(frozen=True)
class Calibration:
    intrinsic: np.ndarray
    distortion: np.ndarray
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

        return Calibration(intrinsic, dist, rvecs, tvecs, roi)


    @staticmethod
    def find_corners(frame: np.ndarray) -> CornerResult:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        if not ret:
            return None

        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

        return OBJP, corners
    

    def dump(self) -> str:
        return json.dumps({
            "intrinsic_matrix": self.intrinsic.tolist(),
            "distortion_coefficients": self.distortion.tolist(),
            "roi": self.roi,
            "rvecs": [rvec.tolist() for rvec in self.rvecs],
            "tvecs": [tvec.tolist() for tvec in self.tvecs],
        }, indent=2)


def load_video(path: Path) -> np.ndarray:
    cap = cv.VideoCapture(str(path))
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame is not None:
            frames.append(frame)

    cap.release()

    return np.array(frames)


def main(video_path: Path) -> None:
    video = load_video(video_path)

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(Calibration.find_corners, video))

    objpoints = [result[0] for result in results if result is not None]
    imgpoints = [result[1] for result in results if result is not None]

    width, height = video[0].shape[:2]
    calibration = Calibration.calibrate(objpoints, imgpoints, (width, height))

    if calibration:
        sys.stdout.write(calibration.dump())
    else:
        sys.stderr.write('No chessboard corners found in the sampled frames\n')
        sys.exit(1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate a camera using a video of a chessboard')
    parser.add_argument('video', type=Path, help='Path to the video file')
    args = parser.parse_args()

    main(args.video)