import argparse
import sys
import warnings
from pathlib import Path

from get_data import read_params, setup_logging

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

warnings.filterwarnings("ignore")


def load_and_save(config_path, logger):
    """Loads data, prints a tabulated view, and saves it to a specified path."""
    logger.info("Start loading and Saving data...")
    config = read_params(config_path)
    logger.info("Finish loading...")


def main(config_path):
    logger = setup_logging()
    load_and_save(config_path, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
