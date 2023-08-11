
# AI-Content-Detector

This repo contains a AI Content Detector.

## Architecture
![alt text][logo]

[logo]: docs/pipeline.png "pipeline"

## Installation

```bash
pip install -r requirements.txt
```

## Requirements
### Requirement #1
please see the file for the illustration


### Requirement #2
I design the AI content detector by the following steps:

1. For AI content dataset, please see ```data_process.py```. I create my own dataset just for this project. I modify the huggingface dataset ''**yelp_review_full**'', and I change the label of each review based on their review score: if the score is over 2, I set it as machine-generated content and change the score to 0; if the score is less than 2, I set it as human-generated content and change the score to 1.
2. For faster training, validation, and testing, I randomly select 1000 samples for training, 100 for validation, and 100 for testing.
3. I use tokenizer to tokenize the texts and get the **input_ids** and **attention_mask** for model input.
4. For model construction, please see ```model.py```. I use Roberta model and add a dropout layer and a fully connected layer for the binary classification task.
5. For model training process. please see ```train.py```. Dataloaders for training and validation are created. Validation can be done during training, and we save the model checkpoint when the lowest validation loss is achieved.
6. For local model inference and deployment, I use a HTML template ```templates/my_template.html``` with Flask, please run:
    ```bash
    python app.py
    ```
    Or you can use [Postman](https://www.postman.com/) and the provided json file here: ```postman/query.postman_collection.json```

7. For local training, please run:
    ```bash
    sh train.sh
    ```

### Requirement #3
Unit tests are implemented for all critical files, including ```data_process.py```, ```model.py```, ```train.py```, ```utils.py```, and ```app.py```.
For testing, please run:
```bash
pytest tests
```

### Requirement #4
I created 3 docker images for train/inference/lambda, and pushed them to AWS ECR (private repo).

To build and push the docker for SageMaker training, please run:
 ```bash
sh build_and_push_train.sh
```
To build and push the docker for SageMaker depoylment, please run:
 ```bash
sh build_and_push_inference.sh
```
To build and push the docker for AWS Lambda, please run:
 ```bash
sh build_and_push_lambda.sh
```

### Requirement #5
For model training and registry and hosting on SageMaker, please run: 
 ```bash
cd ai_detection
python pipeline_train.py
```

### Requirement #6
Currently I give two ways for API endpoint generation:

1. Use AWS Lambda serverless inference from Docker image and create an endpoint to check the service: https://tls6grkoj4ltb37gbyjmdo3nsy0lamax.lambda-url.ap-southeast-2.on.aws/

    This way is low-cost but slow to response.

2. Use SageMaker to create an endpoint, please run:
    ```bash
    cd ai_detection
    python step_deploy.py
    ```
    I also use AWS Lambda for public access service (to simulate API gateway): https://oxg7c6pr4ckfjoswnkulbah3g40tdzbg.lambda-url.ap-southeast-2.on.aws/


    This way is high-cost but fast to response.

### Requirement #7
Architecture provision has not been completed.
I have completed the network part (```network.tf```).
The basic terraform setup is done (see ```local.tf```, ```variable.tf```, and ```providers.tf```)
I havent finished main part (```main.tf```) about secruity and access management, and storage (```storage.tf```).


## Execution
Training Docker and Pipeline

```bash
chmod +x build_and_push_train.sh
./build_and_push_train
python ai_detection/pipeline_train.py
```

## Note
Results can be viewed on SageMaker Console, I can assign you a role to view my results.