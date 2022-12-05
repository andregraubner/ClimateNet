from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config

import argparse
from os import path


# Parse script input to define data and model paths
parser = argparse.ArgumentParser(description='Train a CGNet model on ClimateNet data.')
parser.add_argument('-m', '--model_path', type=str, help='path to the model directory')
parser.add_argument('-d', '--data_path', type=str, help='path to the data directory')
parser.add_argument('-l', '--level', type=int, help='level of details (min: 1, default: 3)')

args = parser.parse_args()

if args.model_path is None:
    print("Error: must provide a path to the model directory.")
    exit
else:
    model_path = args.model_path

if args.data_path is None:
    print("Error: must provide a path to the data.")
    exit
else:
    data_path = args.data_path

# Load model architecture and hyperparameters 
config = Config(model_path + 'config.json')
cgnet = CGNet(config)

# Load input data
train = ClimateDatasetLabeled(path.join(data_path, 'train'), config)

# Print model
if args.level is not None:
    cgnet.print_model(train, args.level)
else:
    cgnet.print_model(train)