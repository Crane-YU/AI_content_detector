resource "aws_iam_role" "sagemaker_role" {
  name = "sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Principal = {
          Service = "sagemaker.amazonaws.com"
        },
        Effect = "Allow"
      }
    ]
  })
  
  # Managed policy attachment
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
  ]
}

resource "aws_iam_role_policy" "invoke_sagemaker_runtime_endpoint" {
  name = "invoke-sagemaker-runtime-endpoint"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = ["s3:ListBucket"],
        Effect   = "Allow",
        Resource = ["arn:aws:s3:::SageMaker"]
      },
      {
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
        Effect   = "Allow",
        Resource = ["arn:aws:s3:::SageMaker/*"]
      },
      {
        Sid      = "Statement1",
        Effect   = "Allow",
        Action   = ["ecr:*"],
        Resource = ["*"]
      }
    ]
  })
}