from typing import Self
from enum import Enum

import open3d as o3d
import numpy as np
import cv2 as cv
import torch
import json


PRED_WIDTH = 640
PRED_HEIGHT = 480


class Model(Enum):
    Small  = 'MiDaS_small'
    Large  = 'DPT_Large'
    Hybrid = 'DPT_Hybrid'

    
    def __str__(self) -> str:
        return self.value


    @staticmethod
    def from_name(name: str) -> Self:
        return Model[name.capitalize()]


def init(type: Model) -> (torch.device, torch.nn.Module, torch.nn.Transformer):
    assert torch.cuda.is_available()

    device = torch.device('cuda:0')
    model = torch.hub.load('intel-isl/MiDaS', str(type), trust_repo=True)
    transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)

    match type:
        case Model.Large | Model.Hybrid:
            return device, model.to(device).eval(), transform.dpt_transform
        
        case Model.Small:
            return device, model.to(device).eval(), transform.small_transform
        

def estimate_depth(img, device, model, transform):
    input = transform(img).to(device)

    with torch.no_grad():
        pred = model(input)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()

    return pred.cpu().numpy()


def depth_to_point_cloud(depth, intrinsic):
    width, height = depth.shape
    depth = depth.astype(np.float32)
    depth = o3d.geometry.Image(depth)

    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    o3d_intrinsic.set_intrinsics(
        height, width,
        intrinsic[0][0], intrinsic[1][1],
        intrinsic[0][2], intrinsic[1][2],
    )

    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, o3d_intrinsic)
    
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    pcd.transform([[-1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
                   
    return pcd


def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])


def normalize(depth):
    norm = cv.normalize(
        depth,
        None,
        0,
        1,
        norm_type=cv.NORM_MINMAX,
        dtype=cv.CV_32F
    )
    norm = (norm * 255.0).astype(np.uint8)
    norm = cv.applyColorMap(norm, cv.COLORMAP_MAGMA)

    return norm


def visualize_depth(results):
    for result in results:
        cv.imshow('Depth', result)
        cv.waitKey(0)
        cv.destroyAllWindows()


def load_image(path: str):
    img = cv.imread(path)
    img = cv.resize(img, (PRED_WIDTH, PRED_HEIGHT), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    return img


def load_intrinsics(path: str):
    with open(path, 'r') as file:
        data = json.load(file)
        matrix = np.array(data)

    return matrix


def dump_depth(depth, path: str):
    np.save(path, depth)


def dump_point_cloud(pcd, path: str):
    o3d.io.write_point_cloud(path, pcd)