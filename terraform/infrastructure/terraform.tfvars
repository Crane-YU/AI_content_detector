## Change project_name to your project name
project_name = "ai-content-detector-terraform" //put your project name here
region = "ap-southeast-2" //change region if desired to deploy in another region

## Change instance types amd volume size for SageMaker if desired
training_instance_type = "ml.m4.xlarge"
inference_instance_type = "ml.m4.xlarge"
volume_size_sagemaker = 30

## Should not be changed with the current folder structure
handler_path  = "../../lambda_function"
handler       = "config_lambda.lambda_handler"

