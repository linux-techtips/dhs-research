from depth import ModelTriple, Model, normalize, estimate
from calibrate import Capture, stream
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import numpy as np
import cv2 as cv

from open3d.pipelines.registration import (
    CorrespondenceCheckerBasedOnEdgeLength,
    CorrespondenceCheckerBasedOnDistance,
    TransformationEstimationPointToPoint,
    FastGlobalRegistrationOption,
    RANSACConvergenceCriteria,
    RegistrationResult,
    Feature,
    registration_ransac_based_on_feature_matching,
    registration_fgr_based_on_feature_matching,
    compute_fpfh_feature,
)
from open3d.geometry import KDTreeSearchParamHybrid, PointCloud, RGBDImage, Image
from open3d.camera import PinholeCameraIntrinsicParameters, PinholeCameraIntrinsic
from open3d.visualization import Visualizer


@dataclass
class Fragment:
    pcd: PointCloud
    feature: Feature


@dataclass(frozen=True)
class DepthImage:
    color: Image
    depth: Image


    @staticmethod
    def estimate(model: ModelTriple, frame: np.ndarray) -> 'DepthImage':
        color_frame = cv.resize(frame, (640, 480))

        cv.imshow('frame', color_frame)

        color_frame = cv.flip(color_frame, 0)

        depth_frame = estimate(model, [color_frame])

        cv.imshow('depth', cv.flip(normalize(depth_frame), 0))

        return DepthImage(Image(color_frame), Image(depth_frame))


    def rgbd(self) -> RGBDImage:
        return RGBDImage.create_from_color_and_depth(self.color, self.depth, convert_rgb_to_intensity=False)


    def fragment(self, voxel_size: float) -> Fragment:
        intrin = PinholeCameraIntrinsic(PinholeCameraIntrinsicParameters.PrimeSenseDefault)

        pcd = PointCloud.create_from_rgbd_image(self.rgbd(), intrin)

        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        radius_norm = voxel_size * 2
        pcd.estimate_normals(search_param=KDTreeSearchParamHybrid(radius=radius_norm, max_nn=50))
        
        radius_feat = voxel_size * 5
        feature = compute_fpfh_feature(pcd, KDTreeSearchParamHybrid(radius=radius_feat, max_nn=120))

        return Fragment(pcd, feature)


@contextmanager
def Visualization(*args, **kwargs) -> Generator[Visualizer, None, None]:
    vis = Visualizer()
    vis.create_window(*args, **kwargs)


# GET ERROR DATA
    try:
        yield vis

    finally:
        vis.destroy_window()


def fast_registration(source: Fragment, target: Fragment, voxel_size: float) -> RegistrationResult:
    return registration_fgr_based_on_feature_matching(
        source.pcd, target.pcd, source.feature, target.feature,
        FastGlobalRegistrationOption(maximum_correspondence_distance=voxel_size * 0.5))


def meh_registration(source: Fragment, target: Fragment, voxel_size: float) -> RegistrationResult:
    distance_thresh = voxel_size * 1.5
    
    return registration_ransac_based_on_feature_matching(
        source.pcd, target.pcd, source.feature, target.feature, True, distance_thresh,
        TransformationEstimationPointToPoint(False), 3, [
            CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(distance_thresh)
        ], RANSACConvergenceCriteria(100_000, 0.999)
    )


def update(vis: Visualizer, cap: cv.VideoCapture) -> None:
    model = ModelTriple.from_model(Model.Hybrid)
    voxel_size = 0.03
    state = None

    for frame in stream(cap):
        if not vis.poll_events():
            break

        frag = DepthImage.estimate(model, frame).fragment(voxel_size)

        if not state:
            state = frag
            vis.add_geometry(state.pcd)
            vis.update_renderer()

            continue

        result = fast_registration(state, frag, voxel_size)
        frag.pcd.transform(result.transformation)

        state.pcd += frag.pcd
        
        vis.update_geometry(state.pcd)
        vis.update_renderer()


def main() -> None:
    with Visualization(width=1280, height=720) as vis, Capture('../data/classroom.mp4') as cap:
        update(vis, cap)

        while vis.poll_events():
            vis.update_renderer()


if __name__ =='__main__':
    main()