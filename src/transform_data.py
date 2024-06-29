import argparse
import sys
import warnings
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))
from get_data import read_params, setup_logging

warnings.filterwarnings("ignore")


def transform_data(config_path):
    """Transform Data."""
    config = read_params(config_path)
    # Add your data transformation logic here


def main(config_path):
    logger = setup_logging()
    transform_data(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
