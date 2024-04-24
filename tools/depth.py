from dataclasses import dataclass
from numpy.typing import NDArray

import numpy as np
import cv2 as cv
import torch


@dataclass(frozen=True)
class MiDas:
    device: torch.device
    model: torch.nn.Module
    transform: torch.nn.Transformer


    @staticmethod
    def Small() -> 'MiDas':
        return MiDas.from_str('MiDaS_small')


    @staticmethod
    def Large() -> 'MiDas':
        return MiDas.from_str('DPT_Large')


    @staticmethod
    def Hybrid() -> 'MiDas':
        return MiDas.from_str('DPT_Hybrid')


    @staticmethod
    def from_str(model_name: str) -> 'MiDas':
        assert torch.cuda.is_available()

        device = torch.device('cuda:0')
        model = torch.hub.load('intel-isl/MiDaS', model_name, trust_repo=True)
        model = model.to(device).eval()
        
        transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
        transform = transform.dpt_transform if model in {'DPT_Large', 'DPT_Hybrid'} else transform.small_transform

        return MiDas(device, model, transform)


<<<<<<< HEAD
def estimate(triple: ModelTriple, batch: np.ndarray) -> np.ndarray:
    input = torch.cat([triple.transform(img) for img in batch]).to(triple.device)

    with torch.no_grad():
        pred = triple.model(input)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=batch[0].shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()
=======
    def estimate(self, batch: list[NDArray[np.float32]]) -> NDArray[np.float32]:
        input = torch.cat([self.transform(img) for img in batch]).to(self.device)
        
        with torch.no_grad():
            pred = self.model(input)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=batch[0].shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()
>>>>>>> cddbfcb429f2812dda420cd9b0e44548be8c122d

        pred = pred.cpu().numpy()
        return pred


def normalize(depth: NDArray[np.float32]) -> NDArray[np.uint8]:
    norm = cv.normalize(depth, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    norm = (norm * 255).astype(np.uint8)
<<<<<<< HEAD
    return cv.applyColorMap(norm, cv.COLORMAP_MAGMA)


if __name__ == '__main__':
    calibration = Calibration.load(Path('../data/pixel_calibration.json'))
    triple = ModelTriple.from_model(Model.Small)

    img = cv.imread('../data/classroom.jpg')
    img = cv.resize(img, (640, 480))

    depth = estimate([img], triple)
    
    with open('../data/depth.npy', 'wb') as file:
        np.save(file, depth)

    depth = normalize(depth)

    cv.namedWindow('Depth', cv.WINDOW_NORMAL)
    cv.moveWindow('Depth', 0, 0)
    cv.imshow('Depth', depth)
    cv.waitKey()

    cv.destroyAllWindows()
=======
    return cv.applyColorMap(norm, cv.COLORMAP_MAGMA)
>>>>>>> cddbfcb429f2812dda420cd9b0e44548be8c122d
