from sagemaker.estimator import Estimator
from sagemaker.inputs import CreateModelInput
from sagemaker.model import Model
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import CreateModelStep, TrainingStep

# from sagemaker.workflow.steps import EndpointConfigStep, CreateEndpointStep

SAGEMAKER_ROLE = "arn:aws:iam::282095691823:role/sagemaker-role"

ecr_docker_training = (
    "282095691823.dkr.ecr.ap-southeast-2.amazonaws.com/detector-training:latest"
)
ecr_docker_inference = (
    "282095691823.dkr.ecr.ap-southeast-2.amazonaws.com/detector-inference:latest"
)

pipeline_session = PipelineSession()


# saved_models
hyperparameters = {
    "train_batch_size": 8,
    "test_batch_size": 8,
    "lr": 5e-5,
    "seed": 42,
    "dropout_rate": 0.3,
    "saved_model_dir": "/opt/ml/model",
    "datafolder": "data",
    "output_path": "/opt/ml/model",
    "epochs": 1,
}

print("--------------model_training_step----------------------")
estimator = Estimator(
    image_uri=ecr_docker_training,
    role=SAGEMAKER_ROLE,
    base_job_name="ai_detection",
    instance_count=1,
    instance_type="ml.m4.xlarge",
    hyperparameters={},
    sagemaker_session=pipeline_session,
)


model_training_step = TrainingStep(
    name="TrainingStep",
    estimator=estimator,
)

print("--------------model_creation_step----------------------")
model = Model(
    image_uri=ecr_docker_inference,
    model_data=model_training_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=SAGEMAKER_ROLE,
    sagemaker_session=pipeline_session,
)

inputs = CreateModelInput(
    instance_type="ml.m4.xlarge",
)
model_creation_step = CreateModelStep(
    name="CreateModelStep",
    model=model,
    inputs=inputs,
)

print("--------------model_registration_step----------------------")

model_approval_status = ParameterString(
    name="ModelApprovalStatus", default_value="PendingManualApproval"
)

model_package_group_name = f"ai-detector-group"

model_registration_step = RegisterModel(
    name="ai-detector",
    content_types=["application/json"],
    response_types=["application/json"],
    estimator=estimator,
    model_data=model_training_step.properties.ModelArtifacts.S3ModelArtifacts,
    model_package_group_name=model_package_group_name,
    approval_status=model_approval_status,
)


# endpoint_config_step = EndpointConfigStep(
#     'EndpointConfigStep',
#     endpoint_config_name=sagemaker.workflow.Pipeline().name + '-endpoint-config',
#     model_name=model_creation_step.properties.ModelName,
#     initial_instance_count=1,
#     instance_type='ml.m4.xlarge'
# )

# create_endpoint_step = CreateEndpointStep(
#     "CreateEndpointStep",
#     endpoint_name=sagemaker.workflow.Pipeline().name + '-endpoint',
#     endpoint_config_name=endpoint_config_step.properties.EndpointConfigName
# )


print("--------------ai-detector-pipeline----------------------")
pipeline_name = f"ai-detector-pipeline"
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        model_approval_status,
    ],
    steps=[model_training_step, model_creation_step, model_registration_step],
)


pipeline.upsert(role_arn=SAGEMAKER_ROLE)
execution = pipeline.start()

print("model is undergoing training...")
