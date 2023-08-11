import os
import sys

sys.path.append(os.getcwd())
from unittest.mock import MagicMock, Mock, patch

import pytest

from data_process import (
    create_small_dataset,
    dataset_label_change,
    load_huggingface_dataset,
    make_fake_label,
    save_dataset,
    tokenize_func,
)

# Mocked dataset example
mocked_dataset = Mock()
mocked_dataset.map = MagicMock(return_value=mocked_dataset)
mocked_dataset.shuffle = MagicMock(return_value=mocked_dataset)
mocked_dataset.select = MagicMock(return_value=mocked_dataset)
mocked_dataset.save_to_disk = MagicMock()


class TestDataProcess:
    def test_load_huggingface_dataset(self):
        with patch("data_process.load_dataset", return_value=mocked_dataset):
            train, test, val = load_huggingface_dataset()
        assert train
        assert test
        assert val

    def test_make_fake_label(self):
        assert make_fake_label({"label": 1}) == {"label": 1}
        assert make_fake_label({"label": 3}) == {"label": 0}

    def test_dataset_label_change(self):
        with patch("data_process.load_dataset", return_value=mocked_dataset):
            train, test, val = dataset_label_change(
                mocked_dataset, mocked_dataset, mocked_dataset
            )
        assert train.map.called
        assert test.map.called
        assert val.map.called

    def test_create_small_dataset(self):
        args = Mock()
        args.seed = 42
        train, test, val = create_small_dataset(
            args, mocked_dataset, mocked_dataset, mocked_dataset
        )
        assert train.select.called_with(range(1000))
        assert test.select.called_with(range(100))
        assert val.select.called_with(range(100))

    def test_tokenize_func(self):
        args = Mock()
        args.saved_model_dir = "roberta-base"
        example = {"text": "Hello world"}
        with patch(
            "data_process.RobertaTokenizer.from_pretrained", return_value=Mock()
        ):
            result = tokenize_func(args, example)
        assert result

    def test_save_dataset(self, tmp_path):
        args = Mock()
        args.datafolder = tmp_path
        save_dataset(args, mocked_dataset, mocked_dataset, mocked_dataset)
        assert mocked_dataset.save_to_disk.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__])
