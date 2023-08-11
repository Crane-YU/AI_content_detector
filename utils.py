import argparse
import json

import torch


def parse_arge():
    """Load and parse the arguments from a json file."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="roberta_config/hyperparameters.json",
        help="Config file",
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            data = json.load(f)

        for key in data:
            args.__dict__[key] = data[key]

    return args


# save the model checkpoint to path
def save_checkpoint(path, model):
    torch.save({"model_state_dict": model.state_dict()}, path)


# load the model checkpoint from path
def load_checkpoint(path, model):
    # load the model to cpu first to increase model loading speed
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict["model_state_dict"])
    return model


# save the train and val metrics to path
def save_metrics(path, train_loss_list, valid_loss_list=None, best_valid_loss=None):
    state_dict = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "best_valid_loss": best_valid_loss,
    }
    torch.save(state_dict, path)


# load the train and val metrics from path
def load_metrics(path, device):
    state_dict = torch.load(path, map_location=device)
    return (
        state_dict["train_loss_list"],
        state_dict["valid_loss_list"],
        state_dict["best_valid_loss"],
    )
