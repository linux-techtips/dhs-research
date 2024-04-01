from calibrate import Capture, stream
from depth import MiDas, normalize

from open3d.camera import PinholeCameraIntrinsicParameters, PinholeCameraIntrinsic

from open3d.geometry import KDTreeSearchParamHybrid, PointCloud, RGBDImage, Image
from open3d.visualization import VisualizerWithKeyCallback as Visualizer
from open3d.pipelines.registration import (
    TransformationEstimationForColoredICP,
    ICPConvergenceCriteria,
    RegistrationResult,
    registration_colored_icp,
)

from contextlib import contextmanager
from typing import Generator

import numpy as np
import cv2 as cv


PRIME_SENSE = PinholeCameraIntrinsic(PinholeCameraIntrinsicParameters.PrimeSenseDefault)


def estimate_rgbd(model: MiDas, frame: np.ndarray) -> RGBDImage:
    depth = model.estimate([frame])

    cv.imshow('frame', frame)
    cv.imshow('depth', normalize(depth))
    
    color = Image(frame)
    depth = Image(depth)

    return RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)


def make_fragment(rgbd: RGBDImage, intrin: PinholeCameraIntrinsic, voxel_size: float) -> PointCloud:
    pcd = PointCloud.create_from_rgbd_image(rgbd, intrin)
    pcd = pcd.voxel_down_sample(voxel_size)

    pcd.estimate_normals(search_param=KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcd.transform([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ])

    return pcd


def colored_registration(source: PointCloud, target: PointCloud, eye: float, voxel_size: float) -> RegistrationResult:
    return registration_colored_icp(
        source, target, voxel_size, eye,
        TransformationEstimationForColoredICP(),
        ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100)
    )

@contextmanager
def Visualize(*args, **kwargs) -> Generator[Visualizer, None, None]:
    vis = Visualizer()
    vis.create_window(*args, **kwargs)

    try:
        yield vis

    finally:
        vis.destroy_window()


def main() -> None:
    model = MiDas.Small()
    pause = False

    def callback(_) -> None:
        nonlocal pause
        pause = not pause

    with Visualize(width=640, height=480) as vis, Capture(0) as cap:        
        vis.register_key_callback(ord(' '), lambda _: callback)
        
        eye = np.identity(4)
        voxel_size = 0.05
        it = 0
        
        frames = stream(cap)

        rgbd = estimate_rgbd(model, next(frames))
        pcd = make_fragment(rgbd, PRIME_SENSE, voxel_size)

        vis.add_geometry(pcd)
        vis.update_renderer()

        for frame in frames:
            if not vis.poll_events():
                break

            it += 1
            if it % 5 != 0 or pause:
                vis.update_renderer()
                continue
            
            rgbd = estimate_rgbd(model, frame)
            frag = make_fragment(rgbd, PRIME_SENSE, voxel_size)

            result = colored_registration(pcd, frag, eye, voxel_size)
            pcd.transform(result.transformation)
            eye = result.transformation

            pcd += frag

            vis.remove_geometry(pcd)
            pcd = pcd.voxel_down_sample(0.01)
            vis.add_geometry(pcd)

            vis.update_renderer()


if __name__ == '__main__':
    main()