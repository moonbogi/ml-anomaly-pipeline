#!/bin/bash
set -e

# Install Docker
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker

# Authenticate to ECR
aws ecr get-login-password --region ${aws_region} \
  | docker login --username AWS --password-stdin ${ecr_repo}

# Pull and run the API container
docker pull ${ecr_repo}:latest

docker run -d \
  --restart unless-stopped \
  -p 8000:8000 \
  -e MODEL_NAME=${model_name} \
  -e MODEL_VERSION=${model_version} \
  -e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
  --name anomaly-api \
  ${ecr_repo}:latest
