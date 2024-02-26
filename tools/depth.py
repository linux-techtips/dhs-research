from calibrate import Calibration
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from enum import Enum

import numpy as np
import cv2 as cv
import torch


class Model(Enum):
    Small = 'MiDaS_small'
    Large = 'DPT_Large'
    Hybrid = 'DPT_Hybrid'


    def __str__(self) -> str:
        return self.value


    @staticmethod
    def from_name(name: str) -> Self:
        return Model[name.capitalize()]
    

@dataclass(frozen=True)
class ModelTriple:
    device: torch.device
    model: torch.nn.Module
    transform: torch.nn.Transformer


    @staticmethod
    def from_model(model: Model) -> Self:
        assert torch.cuda.is_available()

        device = torch.device('cuda:0')
        model = torch.hub.load('intel-isl/MiDaS', str(model), trust_repo=True)
        transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
        transform = transform.dpt_transform if model in {Model.Large, Model.Hybrid} else transform.small_transform

        return ModelTriple(device, model.to(device).eval(), transform)


def estimate(batch: np.ndarray, triple: ModelTriple) -> np.ndarray:
    input = torch.cat([triple.transform(img) for img in batch]).to(triple.device)
    
    with torch.no_grad():
        pred = triple.model(input)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=batch[0].shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()

    pred = pred.cpu().numpy() # Try and return pred.cpu().numpy(). Go for it
    return pred


def normalize(depth: np.ndarray) -> np.ndarray:
    norm = cv.normalize(depth, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    norm = (norm * 255).astype(np.uint8)
    return cv.applyColorMap(norm, cv.COLORMAP_MAGMA)


if __name__ == '__main__':
    calibration = Calibration.load(Path('../data/pixel_calibration.json'))
    triple = ModelTriple.from_model(Model.Small)

    print(calibration.error())

    img = cv.imread('../data/classroom.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.imshow('Original', img)

    img = cv.undistort(img, calibration.matrix, calibration.distortion, None, calibration.intrinsic)
    cv.imshow('Undistorted', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    x, y, w, h = calibration.roi
    img = img[y:y+h, x:x+w]
    cv.imshow('Ropenoi', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    depth = estimate(np.array([img]), triple)
    norm = normalize(depth)

    # Mask out the ROI
    roi = calibration.roi

    cv.imshow('Depth', norm)
    cv.waitKey(0)
    cv.destroyAllWindows()