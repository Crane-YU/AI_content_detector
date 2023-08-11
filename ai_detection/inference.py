import logging
import torch
import json
# from transformers import (
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
#     pipeline
# )
import os
import torch
from model import ROBERTAClassifier
from transformers import RobertaTokenizer
from utils import parse_arge, load_checkpoint


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


def model_fn(model_dir="/opt/ml/model"):
    # Load model
    logger.info(f"loading model from {model_dir}")
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)

    # get the model
    model = ROBERTAClassifier(
        n_classes=2, dropout_rate=0.3, model_path=model_dir
    )
    return model, tokenizer


def predict_fn(data, model_and_tokenizer):
    logger.info(f"predict called with {data} ")
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer
    try: 
        input_text = data.pop("inputs", data)

        args = parse_arge()
        
        out = tokenizer(input_text, padding="max_length", truncation=True)

        data = out["input_ids"]
        mask = out["attention_mask"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # transfer to tensor
        data = torch.tensor(data, dtype=torch.long, device=device).view(1, -1)
        mask = torch.tensor(mask, dtype=torch.long, device=device).view(1, -1)

        

        load_model_fp = os.path.join(args.output_path, "model.pkl")
        model = load_checkpoint(load_model_fp, model)
        pred_label = evaluate(model=model, device=device, data=data, mask=mask)

        if pred_label == 0:
            output = "machine generated"
        else:
            output = "human generated"

        result = output

    except Exception as e:
        result = {"error": str(e)}

    return result


def output_fn(prediction, response_content_type):
    logger.info(f'showing outputs...{prediction}')
    result = {
        "prediction": prediction
    }
    result_json = json.dumps(result)

    return result_json

def input_fn(request_body, request_content_type):
    logger.info(f"request_body: {request_body}")
    request = json.loads(request_body)
    
    return request

