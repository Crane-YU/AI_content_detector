resource "aws_s3_bucket_public_access_block" "model_s3" {
    bucket = aws_s3_bucket.model_s3.id

    block_public_acls       = true
    block_public_policy     = true
    ignore_public_acls      = true
    restrict_public_buckets = true
}

resource "aws_s3_bucket" "model_s3" {
    bucket = "${var.environment}-model"
    acl    = "private"

    versioning {
        enabled    = true
    }

    # S3 lifecycle policy
    lifecycle_rule {
        enabled = true
        transition {
            days          = 30
            storage_class = "INTELLIGENT_TIERING"
        }
    }

    tags = local.tags
}

resource "aws_s3_bucket_policy" "model_s3" {
    bucket = aws_s3_bucket.model_s3.id
    policy = templatefile(
        "${path.module}/../config/s3_bucket_policy.json",
        {
            bucket_arn = aws_s3_bucket.model_s3.arn,
            principal_arn = aws_iam_role.ecs_task_role.arn
        }
    )
    depends_on = [
        aws_s3_bucket_public_access_block.model_s3,
        aws_s3_bucket_ownership_controls.model_s3
    ]
}

resource "aws_s3_bucket_ownership_controls" "model_s3" {
    bucket = aws_s3_bucket.model_s3.id
    rule {
        object_ownership = "BucketOwnerPreferred"
    }
    depends_on = [aws_s3_bucket_public_access_block.model_s3]
}
