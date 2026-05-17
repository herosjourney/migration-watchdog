# GCP to AWS Migration Guide

This guide covers migrating your workloads from Google Cloud Platform to Amazon Web Services.

## Creating IAM Roles for Migration

To set up the required IAM permissions for your migration, you need to create an IAM role.
Navigate to the AWS IAM console and click "Create role".
Select "AWS service" as the trusted entity and choose EC2 as the service.
Attach the required policies and create the role with the name "MigrationRole".

This IAM role creation is a repeated operation that must be performed for each environment
(dev, staging, production) and all required values are known in advance.
