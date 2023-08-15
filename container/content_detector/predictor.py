import flask
import io
import json
import os
import torch
from roberta_model import ROBERTAClassifier
from transformers import RobertaTokenizer
from utils import load_checkpoint

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            # define the model
            model = ROBERTAClassifier(
                n_classes=2, dropout_rate=0.3, model_path=model_path
            )
            load_model_fp = os.path.join(model_path, "model.pkl")
            cls.model = load_checkpoint(load_model_fp, model)
        return cls.model

    @classmethod
    def predict(cls, input):
        """
        For the input, do the predictions and return them.

        Args:
            input (text string): The data on which to do the predictions. There will be
                one prediction per input text
        """
        hf_model = "/opt/ml/input/data/training/hf_model"
        tokenizer = RobertaTokenizer.from_pretrained(hf_model)
        out = tokenizer(input, padding="max_length", truncation=True)
        data = out["input_ids"]
        mask = out["attention_mask"]
        # transfer to tensor
        data = torch.tensor(data, dtype=torch.long).view(1, -1)
        mask = torch.tensor(mask, dtype=torch.long).view(1, -1)
        # get the model
        model = cls.get_model()
        # inference
        model.eval()
        with torch.no_grad():
            y_pred = model(input_ids=data, attention_mask=mask)
            pred_label = torch.argmax(y_pred)
        
        pred_label = pred_label.item()
        if pred_label == 0:
            output_text = "machine generated"
        else:
            output_text = "human generated"

        result = output_text
        return result
        

# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    # get the json text input  
    if flask.request.content_type == "application/json":
        raw_data = flask.request.get_json()
        
        data = json.loads(raw_data)
        if "text" not in data:
            data = json.loads(data["body"])
        print(data)
        input_text = data["text"]
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    # Do the prediction
    prediction = ScoringService.predict(input_text)
    print(prediction)
    result = {"prediction": prediction}
    result_json = json.dumps(result)

    return flask.Response(response=result_json, status=200, mimetype="application/json")

