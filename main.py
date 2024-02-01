import open3d as o3d
import pygame as pg
import numpy as np
import asyncio
import cv2
import sys

import model as midas
import tello

MATRIX = np.array([[921.170702, 0.000000, 459.904354],
                   [0.000000, 919.018377, 351.238301],
                   [0.000000, 0.000000, 1.000000]])
DISTORTION = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])
MAX_DEPTH = 100

def point_cloud(depth, matrix):
    width, height = depth.shape    
    depth = depth.astype(np.float32)
    depth = o3d.geometry.Image(depth)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        height, width,
        matrix[0, 0], matrix[1, 1],
        matrix[0, 2], matrix[1, 2],
    )

    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    
    pcd.transform([[-1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    

    o3d.visualization.draw_geometries([pcd])


async def main() -> None:
    device, model, transform = midas.init(midas.Model.Small)
    # screen = pg.display.set_mode((640, 480), pg.DOUBLEBUF)

    for frame in midas.simulate_video('data/classroom.mp4'):
        depth = midas.query(frame, device, model, transform)
        point_cloud(depth, MATRIX)


if __name__ == '__main__':
    # pg.init()
    try:
        asyncio.run(main())

    finally:
        ...
        # pg.quit()