REGION=ap-southeast-2
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ecr_repository_name=ai-content-detector-terraform

chmod +x content_detector/train
chmod +x content_detector/serve

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com
docker build -t ml-training .
docker tag ml-training:latest ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${ecr_repository_name}:latest
docker push ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${ecr_repository_name}