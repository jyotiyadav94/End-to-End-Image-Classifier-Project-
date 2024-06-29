import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
from tabulate import tabulate

sys.path.append(str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")


def setup_logging(log_file="ml_logs.log"):
    """_summary_

    Args:
        log_file (str, optional): _description_. Defaults to 'ml_logs.log'.

    Returns:
        _type_: _description_
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    for handler in handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def read_params(config_path):
    """_summary_

    Args:
        config_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(config_path) as yaml_file:
        return yaml.safe_load(yaml_file)


def print_tabulated_data(data_frame, logger):
    """Print tabulated data frame."""
    logger.info("\n" + tabulate(data_frame.head(), headers="keys", tablefmt="psql"))


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
