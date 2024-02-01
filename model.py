from typing import Generator, Tuple
from dataclasses import dataclass

import torch
import cv2


FOV = 80
WIDTH = 640
HEIGHT = 480


@dataclass(slots=True)
class Model:
    Small  = 'MiDaS_small'
    Large  = 'DPT_Large'
    Hybrid = 'DPT_Hybrid'


def init(model_type: Model) -> Tuple[torch.device, torch.nn.Module, torch.nn.Transformer]:
    assert torch.cuda.is_available()

    device    = torch.device('cuda:0')
    model     = torch.hub.load('intel-isl/MiDaS', model_type, trust_repo=True)
    transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)

    match model_type:
        case Model.Large | Model.Hybrid:
            return device, model.to(device).eval(), transform.dpt_transform
        
        case Model.Small:
            return device, model.to(device).eval(), transform.small_transform


def simulate_video(path: str) -> Generator:
    cap = cv2.VideoCapture(path)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            scale = 44.5
            width = int(frame.shape[1] * scale / 100)
            height = int(frame.shape[0] * scale / 100)

            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame

    finally:
        cap.release()


def query(img, device, model, transform):
    # input_batch = torch.cat([transform(img).to(device) for img in batch])
    input_batch = transform(img).to(device)

    with torch.no_grad():
        pred = model(input_batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()

    return pred.cpu().numpy()