import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.models import resnet50

from src.config import (
    BEST_MODEL_PATH,
    INFERENCE_CONFIG_PATH,
    SELECTED_THRESHOLD_PATH,
    LABEL_MAP_PATH,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def build_resnet50_binary(num_classes: int = 2):
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_inference_assets():
    inference_config = load_json(INFERENCE_CONFIG_PATH)
    threshold_info = load_json(SELECTED_THRESHOLD_PATH)
    label_map = load_json(LABEL_MAP_PATH)
    return inference_config, threshold_info, label_map


def build_transform_from_config(inference_config: dict):
    image_size = inference_config.get("image_size", 224)
    mean = inference_config.get("normalize_mean", [0.485, 0.456, 0.406])
    std = inference_config.get("normalize_std", [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_model():
    device = get_device()
    inference_config, threshold_info, label_map = load_inference_assets()

    num_classes = inference_config.get("num_classes", 2)
    model = build_resnet50_binary(num_classes=num_classes)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    transform = build_transform_from_config(inference_config)

    return {
        "model": model,
        "device": device,
        "transform": transform,
        "inference_config": inference_config,
        "threshold_info": threshold_info,
        "label_map": label_map,
    }


def prepare_pil_image(image: Image.Image, force_rgb: bool = True):
    if force_rgb:
        return image.convert("RGB")
    return image


@torch.no_grad()
def predict_single_image(image: Image.Image, loaded_assets: dict):
    model = loaded_assets["model"]
    device = loaded_assets["device"]
    transform = loaded_assets["transform"]
    threshold_info = loaded_assets["threshold_info"]
    label_map = loaded_assets["label_map"]

    pil_image = prepare_pil_image(image, force_rgb=True)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)[0]

    prob_non_fractured = float(probs[0].item())
    prob_fractured = float(probs[1].item())

    selected_threshold = float(threshold_info.get("selected_threshold", 0.5))
    predicted_index = int(prob_fractured >= selected_threshold)

    predicted_label = label_map.get(str(predicted_index), str(predicted_index))
    threshold_label = "fractured" if predicted_index == 1 else "non_fractured"

    result = {
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
        "decision_label": threshold_label,
        "prob_fractured": prob_fractured,
        "prob_non_fractured": prob_non_fractured,
        "selected_threshold": selected_threshold,
        "is_fractured_by_threshold": bool(predicted_index == 1),
    }

    return result