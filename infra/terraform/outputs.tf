output "s3_bucket_name" {
  description = "S3 bucket for ML artifacts"
  value       = aws_s3_bucket.ml_artifacts.bucket
}

output "ecr_repository_url" {
  description = "ECR repository URL for pushing the API image"
  value       = aws_ecr_repository.api.repository_url
}
