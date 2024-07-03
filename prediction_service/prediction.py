import json
from pathlib import Path
import timm
import torch
import yaml
from PIL import Image
from marshmallow import ValidationError
from marshmallow import Schema, fields
from timm.data.transforms_factory import create_transform

params_path = "params.yaml"

class InvalidImage(Exception):
    def __init__(self, message="Invalid image format or content"):
        self.message = message
        super().__init__(self.message)


class PredictionError(Exception):
    def __init__(self, message="Error occurred during prediction"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path):
    """Read parameters from the specified YAML config file."""
    with open(config_path) as yaml_file:
        return yaml.safe_load(yaml_file)


def load_model(params_path):
    config = read_params(params_path)
    save_dir = Path(config["model_dir"])
    model_name = config["model_parameters"]["model_name"]
    config_name = config["model_parameters"]["config_name"]
    model_path = save_dir / f"{model_name}.pth"
    config_path = save_dir / f"{config_name}.json"
    # Load model
    model = timm.create_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load config
    with open(config_path, "r") as f:
        model_config = json.load(f)

    return model, model_config


def preprocess_image(image, model_config):
    # Ensure the image is of PIL.Image.Image type
    if not isinstance(image, Image.Image):
        raise InvalidImage

    # Preprocess the image according to model's requirements
    transform = create_transform(**model_config)
    tensor = transform(image).unsqueeze(0)
    return tensor


def get_predictions(tensor, model):
    try:
        with torch.no_grad():
            out = model(tensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        return probabilities
    except Exception as e:
        raise PredictionError(str(e))


def get_top_prediction(probabilities):
    try:
        config = read_params(params_path)
        labels_path = Path(config["reports"]["files"])
        with open(labels_path, "r") as f:
            categories = [s.strip() for s in f.readlines()]
        top_prob, top_id = torch.topk(probabilities, 1)
        top_category = categories[top_id[0]]
        return top_category
    except Exception as e:
        raise PredictionError(str(e))


def response(image, model, model_config):
    try:
        # Preprocess the input image
        tensor = preprocess_image(image, model_config)
        # Get the predictions
        probabilities = get_predictions(tensor, model)
        # Get top prediction (category string)
        result = get_top_prediction(probabilities)
        return result
    except InvalidImage as e:
        return {"error": str(e)}
    except PredictionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": "Unexpected error occurred"}
