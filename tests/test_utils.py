import os
import sys

sys.path.append(os.getcwd())
import json
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from utils import (
    load_checkpoint,
    load_metrics,
    parse_arge,
    save_checkpoint,
    save_metrics,
)

# Constants: will be cleaned
MODEL_PATH = "test_model.pt"
METRICS_PATH = "test_metrics.pt"


# Define a dummy model for testing
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestUtils:
    # Test parse_arge function
    def test_parse_arge_with_config_file(self, tmp_path):
        config_data = {"key1": "value1", "key2": "value2"}

        # Create a temporary json config file
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with patch(
            "argparse.ArgumentParser.parse_args", return_value=Mock(config=config_file)
        ):
            args = parse_arge()
            assert args.key1 == "value1"
            assert args.key2 == "value2"

    def test_parse_arge_without_config_file(self):
        with patch(
            "argparse.ArgumentParser.parse_args", return_value=Mock(config=None)
        ):
            args = parse_arge()
            assert args.config is None

    # Test save_checkpoint and load_checkpoint functions
    def test_save_and_load_checkpoint(self):
        model = DummyModel()
        save_checkpoint(MODEL_PATH, model)

        loaded_model = load_checkpoint(MODEL_PATH, DummyModel())
        for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(param1.data, param2.data)

    # Test save_metrics and load_metrics functions
    def test_save_and_load_metrics(self):
        train_loss = [0.5, 0.4, 0.3]
        val_loss = [0.6, 0.5, 0.4]
        best_val_loss = 0.4
        save_metrics(METRICS_PATH, train_loss, val_loss, best_val_loss)

        train_loss_loaded, val_loss_loaded, best_val_loss_loaded = load_metrics(
            METRICS_PATH, "cpu"
        )
        assert train_loss == train_loss_loaded
        assert val_loss == val_loss_loaded
        assert best_val_loss == best_val_loss_loaded

    @classmethod
    def teardown_class(cls):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(METRICS_PATH):
            os.remove(METRICS_PATH)
        print("Cleaning up after tests!")


if __name__ == "__main__":
    pytest.main([__file__])
