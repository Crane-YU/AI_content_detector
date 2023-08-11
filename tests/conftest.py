import os
import sys

sys.path.append(os.getcwd())
from unittest.mock import MagicMock, Mock

import pytest
import torch

from model import ROBERTAClassifier

# Constants
BATCH_SIZE = 8
SEQUENCE_LENGTH = 256
N_CLASSES = 2


# This fixture sets up a mock dataset and dataloader
@pytest.fixture
def setup_data():
    mocked_dataset = MagicMock()
    mocked_dataset.set_format = MagicMock()
    type(mocked_dataset).__len__ = Mock(return_value=2)
    mocked_dataset.__getitem__.return_value = {
        "input_ids": torch.randint(0, 49395, (BATCH_SIZE, SEQUENCE_LENGTH)),
        "attention_mask": torch.randint(0, 2, (BATCH_SIZE, SEQUENCE_LENGTH)),
        "label": torch.randint(0, N_CLASSES, (BATCH_SIZE,)),
    }

    # Mock DataLoader
    mocked_dataloader = MagicMock()
    type(mocked_dataloader).__len__ = Mock(return_value=1)
    type(mocked_dataloader).__getitem__ = Mock(
        return_value=mocked_dataset.__getitem__()
    )  # Add this line.
    mocked_dataloader.__iter__.return_value = iter(
        [mocked_dataset.__getitem__() for _ in range(10)]
    )

    return mocked_dataset, mocked_dataloader


@pytest.fixture()
def setup_model():
    args = Mock()
    args.epochs = 1
    args.lr = 1e-5
    args.dropout_rate = 0.5
    args.saved_model_dir = "./saved_models"
    args.output_path = "./saved_models"

    model = ROBERTAClassifier(
        n_classes=N_CLASSES,
        dropout_rate=args.dropout_rate,
        model_path=args.saved_model_dir,
    )
    return args, model
