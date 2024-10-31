"""
Script to print the architecture of a PIE model

Usage: `python get_pie_model_params.py model.tar`
"""

import json
import os
import tarfile
import sys

import torch

from pie import utils
from pie.data import MultiLabelEncoder

def load_model(tar_path):
    """Custom function for this notebook, yet in PaPie need to add extra code blocks from the original Encoder.load() method:
    - check tar_path extension
    - check model and current commit values
    - use get_gzip_from_tar function from utils
    """
    with tarfile.open(tar_path) as tar:
        # load label encoder
        label_encoder = MultiLabelEncoder.load_from_string(utils.get_gzip_from_tar(tar, 'label_encoder.zip'))

        # load model parameters
        params = json.loads(utils.get_gzip_from_tar(tar, 'parameters.zip'))

        # load state_dict
        with utils.tmpfile() as tmppath:
            tar.extract('state_dict.pt', path=tmppath)
            dictpath = os.path.join(tmppath, 'state_dict.pt')
            state_dict = torch.load(dictpath, map_location='cpu')

        return label_encoder, params, state_dict

def show_model_architecture(tar_path):
    label_encoder, model_params, state_dict = load_model(tar_path)
    print(f"Label encoder:\n{label_encoder}\n")
    print(f"Model params:\n{model_params}\n")
    print(f"Model layers:\n{list(state_dict.keys())}")
    wemb_dim, cemb_dim, hidden_size, num_layers = model_params["args"]
    print(f"hidden_size: {hidden_size}")
    print(f"num_layers: {num_layers}")
    if "cemb.emb.weight" in state_dict:
        char_vocab_size, char_emb_size = state_dict["cemb.emb.weight"].shape
        assert cemb_dim == char_emb_size, \
            (f"cemb_dim in parameters.zip differs from the actual model cemb_dim: "
             f"{cemb_dim} != {char_emb_size}")
        print(f"cemb_dim: {char_emb_size}")
        print(f"Size of characters vocabulary (incl. 2 extra/default chars): {char_vocab_size}")
    if wemb_dim:
        print(f"wemb_dim: {wemb_dim}")
    print(f"Size of words vocabulary (incl. 2 extra/default words): {len(label_encoder.word)}")
    for task_name in label_encoder.tasks:
        print(f"Number of labels for task {task_name} (incl. 1 extra/default label ??): {len(label_encoder.tasks[task_name].table)}")


if __name__ == "__main__":
    tar_path = sys.argv[1]
    show_model_architecture(tar_path)
