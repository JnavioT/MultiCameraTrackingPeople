from typing import List, Union
import torch
import numpy as np

from tools import log
from tools.preprocessing import create_extractor


STATIC_ATTRIBUTES = {
    "color": ["yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black",
              "purple", "pink"],
    "type": ["sedan", "suv", "van", "hatchback", "mpv",
             "pickup", "bus", "truck", "estate", "sportscar", "RV", "bike"],
}

DYNAMIC_ATTRIBUTES = {
    "brake_signal": ["off", "on"],
}


def get_attribute_value(name: str, value: int):
    """Get the description of an attribute, e.g. get_attribute_value('color', 5) -> 'blue'."""
    if name == "speed":
        return str(value)
    if name in STATIC_ATTRIBUTES:
        return STATIC_ATTRIBUTES[name][value]
    if name in DYNAMIC_ATTRIBUTES:
        return DYNAMIC_ATTRIBUTES[name][value]
    err = f"Invalid static or dynamic attribute name: {name}."
    raise ValueError(err)


def net_is_convolutional(model: torch.nn.Module):
    if isinstance(model, torch.nn.Conv2d):
        return True

    for child in model.children():
        if net_is_convolutional(child):
            return True
    return False


class AttributeExtractor:
    """Base class for extracting dynamic and static attributes from images and re-id features."""

    def __init__(self, models):
        self.models = models
        model = next(iter(models.values()))
        self.device = next(iter(model.parameters())).device
        self.dtype = next(iter(model.parameters())).dtype

        self.attribute_idx = {k: i for i, k in enumerate(self.models.keys())}
        self.num_attributes = len(self.attribute_idx)
        self.attribute_name = {v: k for k, v in self.attribute_idx.items()}

    def __call__(self, X: torch.Tensor, batch_size=1):
        """Computes attributes from image inputs or re-id feature inputs."""
        out = self.run_extract(X, batch_size).cpu().numpy()
        result = {}
        for attrib, idx in self.attribute_idx.items():
            result[attrib] = list(out[:, idx])
        return result

    def run_extract(self, X, batch_size):
        """Extract attributes from X using either CNN or FCNN models."""
        num_samples = X.shape[0]
        X = X.type(self.dtype)
        out = torch.zeros((num_samples, self.num_attributes), dtype=torch.int32,
                          device=self.device)

        for attrib, model in self.models.items():
            attrib_idx = self.attribute_idx[attrib]
            for i in range(0, num_samples, batch_size):
                imax = min(num_samples, i + batch_size)
                X_in = X[i:imax]
                with torch.no_grad():
                    Y = model(X_in.to(self.device))
                    out[i:imax, attrib_idx] = Y.argmax(1)
        return out.to("cpu")


class AttributeExtractorMixed:
    """Computes attributes using FCNN or/and CNN models."""

    def __init__(self, model_paths_by_attribute, fp16=False, device="cuda:0", batch_size=1):
        self.models_reid, self.models_img = {}, {}
        self.batch_size = batch_size

        for name, path in model_paths_by_attribute.items():
            model = torch.load(path)
            model.eval()
            if fp16:
                model.half()
            model.to(device)
            if net_is_convolutional(model):
                self.models_img[name] = model
            else:
                self.models_reid[name] = model
        self.reid_extractor = None if len(
            self.models_reid) == 0 else AttributeExtractor(self.models_reid)
        if len(self.models_img) == 0:
            self.cnn_extractor = None
        else:
            self.cnn_extractor = create_extractor(
                AttributeExtractor, models=self.models_img, batch_size=batch_size)
        log.debug(f"Attribute extractors loaded. Exracted from re-id: {list(self.models_reid.keys())}, "
                  f"Extracted from images: {list(self.models_img.keys())}.")

    def __call__(self, frame: np.ndarray, bboxes: List[Union[List, np.ndarray]], X_reid: torch.Tensor):
        """Computes attributes from image inputs and/or re-id feature inputs."""
        result = {}
        if self.reid_extractor is not None:
            for k, v in self.reid_extractor(X_reid, batch_size=self.batch_size).items():
                result[k] = v
        if self.cnn_extractor is not None:
            for k, v in self.cnn_extractor(frame, bboxes).items():
                result[k] = v
        return result
