resource "aws_security_group" "ecs_security_group" {
    name        = "${var.name} ${var.environment} security group"
    description = "enable http access on port 80"
    vpc_id      = aws_vpc.vpc.id


    ingress {
        description = "http access"
        from_port   = 80
        to_port     = 80
        protocol    = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    egress {
        from_port   = 0
        to_port     = 0
        protocol    = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }

    tags = local.tags
}

resource "aws_security_group_rule" "task_ingress_80" {
    security_group_id        = aws_security_group.ecs_security_group.id
    type                     = "ingress"
    protocol                 = "tcp"
    from_port                = 80
    to_port                  = 80
    source_security_group_id = aws_security_group.ecs_security_group.id
}

resource "aws_vpc" "vpc" {
    cidr_block           = local.vpc_cidr
    instance_tenancy     = "default"
    enable_dns_hostnames = true
    tags                 = local.tags
}

resource "aws_route_table_association" "public" {
    for_each = toset([
        "ap-southeast-2a",
        "ap-southeast-2b"
    ])
    subnet_id      = aws_subnet.public[each.key].id
    route_table_id = aws_route_table.public.id
}

resource "aws_internet_gateway" "igw" {
    vpc_id = aws_vpc.vpc.id
    tags   = local.tags
}


resource "aws_cloudwatch_log_group" "web_api_gateway" {
    name = "/aws/api_gw/${aws_apigatewayv2_api.httpapi.name}"
    retention_in_days = 30
}

resource "aws_route_table" "public" {
    vpc_id = aws_vpc.vpc.id
    tags   = local.tags
}

resource "aws_route" "public_subnet_igw" {
    route_table_id         = aws_route_table.public.id
    destination_cidr_block = "0.0.0.0/0"
    gateway_id             = aws_internet_gateway.igw.id
}

resource "aws_subnet" "public" {
    for_each = {
        "ap-southeast-2a" : cidrsubnet(local.vpc_cidr, 6, 0),
        "ap-southeast-2b" : cidrsubnet(local.vpc_cidr, 6, 1),
    }

    availability_zone = each.key

    vpc_id                  = aws_vpc.vpc.id
    cidr_block              = each.value
    map_public_ip_on_launch = true

    tags   = local.tags
}