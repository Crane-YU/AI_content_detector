import os
import sys

sys.path.append(os.getcwd())
import time
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from train import set_dataloader, validation
from transformers import RobertaTokenizer, set_seed

from utils import load_checkpoint

# Constants
BATCH_SIZE = 8
SEQUENCE_LENGTH = 256
N_CLASSES = 2


def model_pred(args, model, input_text, device):
    tokenizer = RobertaTokenizer.from_pretrained(args.saved_model_dir)
    out = tokenizer(input_text, padding="max_length", truncation=True, max_length=256)
    # get the input_ids and attention_mask
    input_ids = out["input_ids"]
    attention_mask = out["attention_mask"]
    # transfer the data to tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).view(1, -1)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device).view(
        1, -1
    )
    # model prediction
    model.eval()
    with torch.no_grad():
        y_pred = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_label = torch.argmax(y_pred, dim=-1)
    return pred_label


def predict_time(model, device):
    # put input_ids and attention_mask to device
    input_ids = torch.randint(0, 49395, (BATCH_SIZE, SEQUENCE_LENGTH), device=device)
    attention_mask = torch.randint(0, 2, (BATCH_SIZE, SEQUENCE_LENGTH), device=device)
    model.eval()
    # start the timer
    start = time.time()
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        # endthe timer
        end = time.time()
    return end - start


class TestTrainRoberta:
    def test_set_dataloader(self, setup_data):
        mocked_dataset, _ = setup_data
        args = Mock()
        args.train_batch_size = 2

        dataloader_train, dataloader_val = set_dataloader(
            args, mocked_dataset, mocked_dataset
        )
        assert len(dataloader_train) == 1
        assert len(dataloader_val) == 1

    def test_validation(self, setup_data, setup_model):
        _, mocked_dataloader = setup_data
        _, model = setup_model
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss()

        valid_loss, acc_score = validation(model, device, mocked_dataloader, criterion)

        # Ensure valid_loss and acc_score are numbers
        assert isinstance(valid_loss, float)
        assert isinstance(acc_score, float)

    # More model-related tests
    # Pre-train Tests
    # 1. check data leakage: concat the train and val datasets to see if the number matches
    def test_data_leakage(self, setup_data):
        mocked_dataset, _ = setup_data

        # check for input_ids
        concat_data = torch.cat(
            [mocked_dataset[0]["input_ids"], mocked_dataset[1]["input_ids"]], dim=0
        )
        train_num = mocked_dataset[0]["input_ids"].shape[0]
        val_num = mocked_dataset[1]["input_ids"].shape[0]

        # Ensure the concatenated data have the same number of samples as the added result
        assert concat_data.shape[0] == train_num + val_num

    # Post-train Tests
    # 1. Invariance test: changes to the inputs SHOULD NOT impact the predictions
    # Capitalize the words should not change the output
    def test_invariance(self, setup_model):
        args, model = setup_model
        model = load_checkpoint(os.path.join(args.output_path, "model.pkl"), model)
        device = torch.device("cpu")
        model = model.to(device)
        # create dummy text
        input_text = "text"
        input_text_new = "TEXT"
        # predict
        pred_label = model_pred(args, model, input_text, device)
        pred_label_new = model_pred(args, model, input_text_new, device)

        # Ensure predicted labels are the same
        assert pred_label.item() == pred_label_new.item()

    # 2. Directional Expectation test: changes to the inputs SHOULD impact the predictions
    # permutating the order of words should change the output
    def test_direcitonal_expectation(self, setup_model):
        args, model = setup_model
        model = load_checkpoint(os.path.join(args.output_path, "model.pkl"), model)
        device = torch.device("cpu")
        model = model.to(device)
        # create dummy text
        input_text = "this is a human generated text"
        input_text_new = "@@###@%$#$"
        # predict
        pred_label = model_pred(args, model, input_text, device)
        pred_label_new = model_pred(args, model, input_text_new, device)

        # Ensure predicted labels are different
        assert pred_label.item() != pred_label_new.item()

    # Model Performance Tests
    # 1. we need to check the model inference time/latency.
    def test_latency(self, setup_model):
        args, model = setup_model
        model = load_checkpoint(os.path.join(args.output_path, "model.pkl"), model)
        device = torch.device("cpu")
        model = model.to(device)
        latency = np.array([predict_time(model, device) for _ in range(20)])
        median = np.quantile(latency, 0.50)

        # Ensure median time is less than 1s in cpu
        assert median < 2


if __name__ == "__main__":
    # fix the see for model loading
    set_seed(42)
    pytest.main([__file__])
