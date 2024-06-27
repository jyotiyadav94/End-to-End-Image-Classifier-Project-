import os 
import sys
import json
import torch
import argparse
import warnings
import logging 
import pandas as pd
from torch import nn
from get_data import read_params, print_tabulated_data,setup_logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))
warnings.filterwarnings("ignore")

def analyse_data(config_path):
    """Analyse Data."""
    # Add your data Analyse & Visualize logic here

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    analyse_data(config_path=parsed_args.config)