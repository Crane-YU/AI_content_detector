terraform {
    backend "s3" {
        region = "ap-southeast-2"
    }
}

provider "aws" {
    region = var.region
}