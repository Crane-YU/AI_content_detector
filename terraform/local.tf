locals {
    assume_role_policy = <<JSON
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "sts:AssumeRole",
                "Principal": {
                    "Service": [
                        "ecs.amazonaws.com",
                        "events.amazonaws.com",
                        "ecs-tasks.amazonaws.com"
                    ]
                }
            }
        ]
    }
    JSON
    
    vpc_cidr = "10.1.0.0/16"
    tags = {
            "Service" = var.name,
            "Environment" = var.environment,
        }
}