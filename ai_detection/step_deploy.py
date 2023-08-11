import boto3

sagemaker_client = boto3.client("sagemaker")


endpoint_name = "ai-detector-endpoint-8"
model_name = "pipelines-3vfdqp8wa7mf-createmodelstep-clqyd4ur99"


# Create endpoint configuration
endpoint_config_name = endpoint_name + "-config"
sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "InstanceType": "ml.m4.xlarge",
            "InitialInstanceCount": 1,
            "ModelName": model_name,
            "VariantName": "AllTraffic",
        }
    ],
)

# Create endpoint
sagemaker_client.create_endpoint(
    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
)

print("model is undergoing deployment...")