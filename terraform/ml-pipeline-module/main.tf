#################################################
# Variables & Locals
#################################################

terraform {
  required_version = ">= 0.13.5"
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

locals {
  input_training_path = "s3://${var.s3_bucket_input_training_path}"
  output_models_path = "s3://${var.s3_bucket_output_models_path}"
}

#################################################
# IAM Roles and Policies for Step Functions
#################################################

// IAM role for Step Functions state machine
data "aws_iam_policy_document" "sf_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["states.${data.aws_region.current.name}.amazonaws.com"]
    }
  }
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sf_exec_role" {
  name               = "${var.project_name}-sfn-exec"
  assume_role_policy = data.aws_iam_policy_document.sf_assume_role.json
}

// policy to invoke lambda
data "aws_iam_policy_document" "lambda_invoke" {
  statement {
    actions = [
      "lambda:InvokeFunction"
    ]
    resources = [
      aws_lambda_function.lambda_function.arn,
    ]
  }
}

resource "aws_iam_policy" "lambda_invoke" {
  name   = "${var.project_name}-lambda-invoke"
  policy = data.aws_iam_policy_document.lambda_invoke.json
}

resource "aws_iam_role_policy_attachment" "lambda_invoke" {
  role       = aws_iam_role.sf_exec_role.name
  policy_arn = aws_iam_policy.lambda_invoke.arn
}

// policy to invoke sagemaker training job, creating endpoints etc.
resource "aws_iam_policy" "sagemaker_policy" {
  name   = "${var.project_name}-sagemaker"
  policy = <<-EOF
      {
          "Version": "2012-10-17",
          "Statement": [
              {
                  "Effect": "Allow",
                  "Action": [
                      "sagemaker:CreateTrainingJob",
                      "sagemaker:DescribeTrainingJob",
                      "sagemaker:StopTrainingJob",
                      "sagemaker:createModel",
                      "sagemaker:createEndpointConfig",
                      "sagemaker:createEndpoint",
                      "sagemaker:addTags"
                  ],
                  "Resource": [
                   "*"
                  ]
              },
              {
                  "Effect": "Allow",
                  "Action": [
                      "sagemaker:ListTags"
                  ],
                  "Resource": [
                   "*"
                  ]
              },
              {
                  "Effect": "Allow",
                  "Action": [
                      "iam:PassRole"
                  ],
                  "Resource": [
                   "*"
                  ],
                  "Condition": {
                      "StringEquals": {
                          "iam:PassedToService": "sagemaker.amazonaws.com"
                      }
                  }
              },
              {
                  "Effect": "Allow",
                  "Action": [
                      "events:PutTargets",
                      "events:PutRule",
                      "events:DescribeRule"
                  ],
                  "Resource": [
                  "*"
                  ]
              }
          ]
      }
EOF
}

resource "aws_iam_role_policy_attachment" "sm_invoke" {
  role       = aws_iam_role.sf_exec_role.name
  policy_arn = aws_iam_policy.sagemaker_policy.arn
}

resource "aws_iam_role_policy_attachment" "cloud_watch_full_access" {
  role       = aws_iam_role.sf_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchFullAccess"
}

#################################################
# IAM Roles and Policies for SageMaker
#################################################

// IAM role for SageMaker training job
data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_exec_role" {
  name               = "${var.project_name}-sagemaker-exec"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json
}

// Policies for sagemaker execution training job
resource "aws_iam_policy" "sagemaker_s3_policy" {
  name   = "${var.project_name}-sagemaker-s3-policy"
  policy = <<-EOF
      {
          "Version": "2012-10-17",
          "Statement": [
              {
                  "Effect": "Allow",
                  "Action": [
                      "s3:*"
                  ],
                  "Resource": [
                   "${aws_s3_bucket.bucket_training_data.arn}",
                   "${aws_s3_bucket.bucket_output_models.arn}",
                   "${aws_s3_bucket.bucket_training_data.arn}/*",
                   "${aws_s3_bucket.bucket_output_models.arn}/*"
                  ]
              }
          ]
      }
EOF
}

resource "aws_iam_role_policy_attachment" "s3_restricted_access" {
  role       = aws_iam_role.sagemaker_exec_role.name
  policy_arn = aws_iam_policy.sagemaker_s3_policy.arn
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

#################################################
# ECR Repository
#################################################

resource "aws_ecr_repository" "ecr_repository" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = false
  }
}

#################################################
# S3 Buckets
#################################################
resource "aws_s3_bucket" "bucket_training_data" {
  bucket = var.s3_bucket_input_training_path
}

# upload data files and hugginface model files
resource "aws_s3_object" "bucket_training_data_files" {
  for_each = fileset(var.s3_object_training_data, "**")
  bucket = aws_s3_bucket.bucket_training_data.id
  key    = each.key
  source = "${var.s3_object_training_data}/${each.value}"
  etag   = filemd5("${var.s3_object_training_data}/${each.value}")
}

resource "aws_s3_bucket" "bucket_output_models" {
  bucket = var.s3_bucket_output_models_path
}

#################################################
# Lambda
#################################################

resource "aws_iam_role" "lambda-role" {
  name        = "${var.lambda_function_name}-role"
  description = "${var.lambda_function_name}-permissions"
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": ["lambda.amazonaws.com"]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

}

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = var.lambda_folder
  output_path = var.lambda_zip_filename
}

resource "aws_lambda_function" "lambda_function" {

  filename         = data.archive_file.lambda_zip.output_path
  function_name    = var.lambda_function_name
  role             = aws_iam_role.lambda-role.arn
  handler          = var.handler
  runtime          = var.runtime
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  memory_size      = var.memory_size
  timeout          = var.timeout
}


#################################################
# Step Function
#################################################

resource "aws_sfn_state_machine" "sfn_state_machine" {
  name     = "${var.project_name}-state-machine"
  role_arn = aws_iam_role.sf_exec_role.arn

  definition = <<-EOF
  {
  "Comment": "An AWS Step Function State Machine to train, build and deploy an Amazon SageMaker model endpoint",
  "StartAt": "Configuration Lambda",
  "States": {
    "Configuration Lambda": {
      "Type": "Task",
      "Resource": "${aws_lambda_function.lambda_function.arn}",
      "Parameters": {
        "PrefixName": "${var.project_name}",
        "input_training_path": "$.input_training_path"
        },
      "Next": "Create Training Job",
      "ResultPath": "$.training_job_name"
      },
    "Create Training Job": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
      "Parameters": {
        "TrainingJobName.$": "$.training_job_name",
        "ResourceConfig": {
          "InstanceCount": 1,
          "InstanceType": "${var.training_instance_type}",
          "VolumeSizeInGB": ${var.volume_size_sagemaker}
        },
        "HyperParameters": {
            "saved_model_dir": "/opt/ml/input/data/training/hf_model",
            "datafolder": "/opt/ml/input/data/training",
            "output_path": "/opt/ml/model",
            "epochs": "1",
            "train_batch_size": "8",
            "test_batch_size": "8",
            "lr": "0.00005",
            "seed": "42",
            "dropout_rate": "0.3"
        },
        "AlgorithmSpecification": {
          "TrainingImage": "${aws_ecr_repository.ecr_repository.repository_url}",
          "TrainingInputMode": "File"
        },
        "OutputDataConfig": {
          "S3OutputPath": "s3://${aws_s3_bucket.bucket_output_models.bucket}"
        },
        "StoppingCondition": {
          "MaxRuntimeInSeconds": 86400
        },
        "RoleArn": "${aws_iam_role.sagemaker_exec_role.arn}",
        "InputDataConfig": [
        {
          "ChannelName": "training",
          "DataSource": {
            "S3DataSource": {
              "S3DataType": "S3Prefix",
              "S3Uri": "s3://${aws_s3_bucket.bucket_training_data.bucket}",
              "S3DataDistributionType": "FullyReplicated"
            }
          }
        }
        ]
      },
      "Next": "Create Model"
    },
    "Create Model": {
      "Parameters": {
        "PrimaryContainer": {
          "Image": "${aws_ecr_repository.ecr_repository.repository_url}",
          "Environment": {},
          "ModelDataUrl.$": "$.ModelArtifacts.S3ModelArtifacts"
        },
        "ExecutionRoleArn": "${aws_iam_role.sagemaker_exec_role.arn}",
        "ModelName.$": "$.TrainingJobName"
      },
      "Resource": "arn:aws:states:::sagemaker:createModel",
      "Type": "Task",
      "ResultPath":"$.taskresult",
      "Next": "Create Endpoint Config"
    },
    "Create Endpoint Config": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createEndpointConfig",
      "Parameters":{
        "EndpointConfigName.$": "$.TrainingJobName",
        "ProductionVariants": [
        {
          "InitialInstanceCount": 1,
          "InstanceType": "${var.inference_instance_type}",
          "ModelName.$": "$.TrainingJobName",
          "VariantName": "AllTraffic"
        }
        ]
      },
      "ResultPath":"$.taskresult",
      "Next":"Create Endpoint"
    },
    "Create Endpoint":{
      "Type":"Task",
      "Resource":"arn:aws:states:::sagemaker:createEndpoint",
      "Parameters":{
        "EndpointConfigName.$": "$.TrainingJobName",
        "EndpointName.$": "$.TrainingJobName"
      },
      "End": true
      }
    }
  }
  EOF
}

#################################################
# Outputs
#################################################

output "ecr_repository_url" {
  value = aws_ecr_repository.ecr_repository.repository_url
  description = "ECR URL for the Docker Image"
}