# for model implementaion
import os
import traceback

import torch
from flask import Flask, jsonify, render_template, request, url_for
from transformers import RobertaTokenizer

from model import ROBERTAClassifier
from utils import load_checkpoint, parse_arge

app = Flask(__name__)


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


# The code below lets the Flask server respond to browser requests for a favicon
@app.route("/favicon.ico")
def favicon():
    return url_for("static", filename="data:,")


# use my own template for text input box
@app.route("/")
def home():
    return render_template("my_template.html")


@app.route("/health")
def health():
    response_data = {"status": "ok"}
    return jsonify(response_data)


@app.route("/", methods=["POST"])
def predict():
    input_text = request.form["text"]

    args = parse_arge()
    tokenizer = RobertaTokenizer.from_pretrained(args.saved_model_dir)
    out = tokenizer(input_text, padding="max_length", truncation=True)

    data = out["input_ids"]
    mask = out["attention_mask"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transfer to tensor
    data = torch.tensor(data, dtype=torch.long, device=device).view(1, -1)
    mask = torch.tensor(mask, dtype=torch.long, device=device).view(1, -1)

    # define the model
    model = ROBERTAClassifier(
        n_classes=2, dropout_rate=args.dropout_rate, model_path=args.saved_model_dir
    )

    load_model_fp = os.path.join(args.output_path, "model.pkl")
    model = load_checkpoint(load_model_fp, model)
    pred_label = evaluate(model=model, device=device, data=data, mask=mask)

    if pred_label == 0:
        output = "machine generated"
    else:
        output = "human generated"

    return output


@app.route("/predict", methods=["POST"])
def predict_json():
    try:
        input_json = request.get_json()
        input_text = str(input_json["text"])

        args = parse_arge()
        tokenizer = RobertaTokenizer.from_pretrained(args.saved_model_dir)
        out = tokenizer(input_text, padding="max_length", truncation=True)

        data = out["input_ids"]
        mask = out["attention_mask"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # transfer to tensor
        data = torch.tensor(data, dtype=torch.long, device=device).view(1, -1)
        mask = torch.tensor(mask, dtype=torch.long, device=device).view(1, -1)

        # define the model
        model = ROBERTAClassifier(
            n_classes=2, dropout_rate=args.dropout_rate, model_path=args.saved_model_dir
        )

        load_model_fp = os.path.join(args.output_path, "model.pkl")
        model = load_checkpoint(load_model_fp, model)
        pred_label = evaluate(model=model, device=device, data=data, mask=mask)

        if pred_label == 0:
            output_text = "machine generated"
        else:
            output_text = "human generated"

        response_data = {"message": "Success", "input_data": output_text}
        return jsonify(response_data)

    except Exception as e:
        print(e)
        tb = traceback.format_exc()
        error_response = {"error": str(e), "error_msg": str(tb)}

        return (
            jsonify(error_response),
            400,
        )  # Return a JSON error response with a 400 status code


if __name__ == "__main__":
    app.run()
