import os
import sys

sys.path.append(os.getcwd())
import pytest
import torch
from transformers import RobertaConfig, RobertaModel

from model import ROBERTAClassifier

# Constants
BATCH_SIZE = 8
SEQUENCE_LENGTH = 256
N_CLASSES = 2


class TestModel:
    # Mock the pretrained model loading to avoid downloading models during testing
    @pytest.fixture(scope="function", autouse=True)
    def mock_roberta_pretrained(self, monkeypatch):
        def mock_from_pretrained(*args, **kwargs):
            return RobertaModel(RobertaConfig())

        monkeypatch.setattr(RobertaModel, "from_pretrained", mock_from_pretrained)

    # Test ROBERTAClassifier
    def test_roberta_classifier_initialization(self):
        model = ROBERTAClassifier()
        assert isinstance(model, ROBERTAClassifier)

    # Test output tensor shape
    def test_roberta_classifier_forward(self):
        model = ROBERTAClassifier()
        input_ids = torch.randint(0, 49395, (BATCH_SIZE, SEQUENCE_LENGTH))
        attention_mask = torch.randint(0, 2, (BATCH_SIZE, SEQUENCE_LENGTH))
        out = model(input_ids, attention_mask)
        assert out.shape == (BATCH_SIZE, N_CLASSES)


if __name__ == "__main__":
    pytest.main()
