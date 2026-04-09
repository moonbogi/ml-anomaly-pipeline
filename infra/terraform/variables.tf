variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "anomaly-detector"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-west-2"
}

variable "instance_type" {
  description = "EC2 instance type for the serving API"
  type        = string
  default     = "t3.medium"
}
