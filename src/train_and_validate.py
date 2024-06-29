import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import timm
import torch

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))
from get_data import read_params, setup_logging

warnings.filterwarnings("ignore")


def save_model(model, model_path):
    """Save the model to the specified path."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def save_config(model, config_path):
    """Save the model configuration to the specified path."""
    config = timm.data.resolve_data_config({}, model=model)
    with open(config_path, "w") as f:
        json.dump(config, f)
    print(f"Model configuration saved to {config_path}")


def load_saved_model(model_name, model_path):
    """Load the model from the specified path."""
    model = timm.create_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def train_and_evaluate(config_path, logger):
    """Train and evaluate models."""
    config = read_params(config_path)
    save_dir = Path(config["model_dir"])
    model_name = config["model_parameters"]["model_name"]
    config_name = config["model_parameters"]["config_name"]
    model_path = save_dir / f"{model_name}.pth"
    config_path = save_dir / f"{config_name}.json"

    # Create a new model instance (pretrained=True for initialization)
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    # Save the model
    save_model(model, model_path)

    # Load the saved model
    loaded_model = load_saved_model(model_name, model_path)

    # Save the model configuration
    save_config(model, config_path)

    print("Model loaded and saved successfully.")


def main(config_path):
    logger = setup_logging()
    train_and_evaluate(config_path, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
