from pathlib import Path
import torch
import os
#from ultralytics import YOLO

def load_yolo(which):
    """Load a yolo v5 network from local repository. Download the weights there if needed."""

    cwd = Path.cwd()
    yolo_dir = str(Path(__file__).parent.joinpath("yolov5"))
    os.chdir(yolo_dir)
    model = torch.hub.load(yolo_dir, which, source="local")
    os.chdir(str(cwd))
    return model

def load_yolo2(which):
    """Load a yolo v8 network from local repository. Download the weights there if needed."""

    model = 1
    return model
