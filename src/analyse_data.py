import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from get_data import print_tabulated_data, read_params, setup_logging

sys.path.append(str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")


def analyse_data(config_path):
    """_summary_

    Args:
        config_path (_type_): _description_
    """
    # Add your data Analyse & Visualize logic here


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    analyse_data(config_path=parsed_args.config)
