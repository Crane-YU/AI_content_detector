variable "environment" {
    type = string
    description = "Deployment environment"
    default = "dev"
}

variable "region" {
    type = string
    default = "ap-southeast-2"
}

variable "image_tag" {
    type = string
    description = "The tag of the docker image to be deployed"
}

variable "name" {
    type = string
    default = "ai_detection"
}