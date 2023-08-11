#!/usr/bin/env bash

REPO_NAME=detector

VERSION=$(git rev-parse --short HEAD)

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

REGION=ap-southeast-2

REPO_TRAINING=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}-training
REPO_INFERENCE=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}-inference

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com

DOCKER_BUILDKIT=1 docker build  -t ${REPO_INFERENCE}:${VERSION} -f Dockerfile.inference . --build-arg REGION=${REGION}
docker tag ${REPO_INFERENCE}:${VERSION} ${REPO_INFERENCE}:latest

docker push ${REPO_INFERENCE}:${VERSION}
docker push ${REPO_INFERENCE}:latest