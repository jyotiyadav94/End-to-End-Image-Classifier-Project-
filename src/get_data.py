import os
import sys
import yaml
import logging
import argparse
import warnings
import pandas as pd
from tabulate import tabulate
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))
warnings.filterwarnings("ignore")

def setup_logging(log_file='ml_logs.log'):
    """Configure Logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    for handler in handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def read_params(config_path):
    """Read parameters from the specified YAML config file."""
    with open(config_path) as yaml_file:
        return yaml.safe_load(yaml_file)

def print_tabulated_data(data_frame, logger):
    """Print tabulated data frame."""
    logger.info("\n" + tabulate(data_frame.head(), headers='keys', tablefmt='psql'))

def get_data(config_path, logger):
    """Load and combine data from all the CSV files."""
    logger.info("Start Getting the data...")
    config = read_params(config_path)
    data_path = config["data_source"]["source"]
    logger.info("Print the dataframe..")
    logger.info("Finish getting the data...")
    return " "

def main(config_path):
    logger = setup_logging()
    data = get_data(config_path, logger)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)