import json
import logging
import os

import torch
from transformers import RobertaTokenizer

from model import ROBERTAClassifier
from utils import load_checkpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# model evaluation
def evaluate(model, device, data, mask):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        mask = mask.to(device)
        data_test = data.to(device)

        y_pred = model(input_ids=data_test, attention_mask=mask)
        pred_label = torch.argmax(y_pred)
    return pred_label.item()


def input_fn(request_body, request_content_type):
    logger.info(f"request_body: {request_body}")
    request = json.loads(request_body)

    return request


def predict_fn(data, model_and_tokenizer):
    logger.info(f"predict called with {data} ")
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer
    try:
        input_text = data.pop("inputs", data)
        logger.info(f"type of input_text: {type(input_text)}")
        logger.info(f"input_text: {input_text} ")

        args = {"output_path": "/opt/ml/model"}
        logger.info(f"args: {str(args)} ")

        input_text = str(input_text)
        out = tokenizer(input_text, padding="max_length", truncation=True)
        logger.info(f"out: {str(out)} ")

        data = out["input_ids"]
        logger.info(f"data: {str(data)} ")
        mask = out["attention_mask"]
        logger.info(f"mask: {str(mask)} ")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"device: {str(device)} ")

        # transfer to tensor
        data = torch.tensor(data, dtype=torch.long, device=device).view(1, -1)
        logger.info(f"data: {str(data)} ")
        mask = torch.tensor(mask, dtype=torch.long, device=device).view(1, -1)
        logger.info(f"mask: {str(mask)} ")

        load_model_fp = os.path.join(args["output_path"], "model.pkl")
        logger.info(f"load_model_fp: {str(load_model_fp)} ")
        model = load_checkpoint(load_model_fp, model)
        logger.info(f"model: {str(model)} ")
        pred_label = evaluate(model=model, device=device, data=data, mask=mask)
        logger.info(f"pred_label: {str(pred_label)} ")

        if pred_label == 0:
            output_text = "machine generated"
        else:
            output_text = "human generated"

        result = output_text
        logger.info(f"output_text: {str(result)} ")

    except Exception as e:
        result = {"error": str(e)}
        logger.info(f"result: {str(result)} ")

    return result


def model_fn(model_dir="/opt/ml/model"):
    # Load model
    logger.info(f"loading model from {model_dir}")
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)

    # get the model
    model = ROBERTAClassifier(n_classes=2, dropout_rate=0.3, model_path=model_dir)
    logger.info(f"loading model from {model_dir} DONE")
    return model, tokenizer


def output_fn(prediction, response_content_type):
    logger.info(f"showing outputs...{prediction}")
    result = {"prediction": prediction}
    result_json = json.dumps(result)

    return result_json