#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

# chmod +x train
# chmod +x serve

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-east-1 if none defined)
region=$(aws configure get region)
region=${region:-us-east-1}
echo "Region:"
echo ${region}

if [[ $region =~ ^cn.* ]]
then
    fullname="${account}.dkr.ecr.${region}.amazonaws.com.cn/${image}:latest"
else
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
fi

echo "ECR Fullname:"
echo ${fullname}

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin 282095691823.dkr.ecr.ap-southeast-2.amazonaws.com
echo "logged in to AWS ECR"
# $(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR with the full name.
echo "Build Docker:"
docker build  -t ${image} -f Dockerfile.lambda .
echo "Tag Docker:"
docker tag ${image} ${fullname}

# Can comment out for testing
echo "Push Docker:"
docker push ${fullname}