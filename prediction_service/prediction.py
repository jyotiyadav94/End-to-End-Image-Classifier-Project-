import yaml
import os
import timm
import torch
import json
from pathlib import Path
from timm.data.transforms_factory import create_transform
from PIL import Image

params_path = "params.yaml"

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
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    return model, model_config

model, model_config = load_model(params_path)

def preprocess_image(image):
    config = read_params(params_path)
    transform_config = config["transform"]
    transform = create_transform(**transform_config)
    tensor = transform(image).unsqueeze(0)
    return tensor

def get_predictions(tensor):
    with torch.no_grad():
        out = model(tensor)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    return probabilities

def get_top_predictions(probabilities, top_k=5):
    config = read_params(params_path)
    labels_path = Path(config["reports"]["files"])
    with open(labels_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    top_prob, top_catid = torch.topk(probabilities, top_k)
    results = []
    for i in range(top_prob.size(0)):
        results.append({"category": categories[top_catid[i]], "probability": float(top_prob[i])})
    
    return results

def form_response(image):
    # Preprocess the input data
    tensor = preprocess_image(image)
    print(tensor)
    # Get the prediction
    probabilities = get_predictions(tensor)
    print(probabilities)
    # Get top 5 prediction
    results = get_top_predictions(probabilities)
    return results

def api_response(image):
    try:
        # Preprocess the input data
        tensor = preprocess_image(image)
        
        # Get the prediction
        probabilities = get_predictions(tensor)

        # return the results
        results = get_top_predictions(probabilities)
        return results

    except Exception as e:
        return {"error": str(e)}
