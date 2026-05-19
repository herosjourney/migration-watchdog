# GCP to AWS Migration Guide

This guide covers migrating your workloads from Google Cloud Platform to Amazon Web Services.

## Step 1: Configure IAM Permissions

Navigate to the AWS IAM console and create a new IAM role for your migration workload.
Click "Create role", select "AWS service", and follow the wizard to complete the setup.

## Step 2: Request Service Quota Increase

Navigate to the AWS Service Quotas console and search for "Amazon EC2".
Select the "Running On-Demand Standard instances" quota and click "Request quota increase".
Enter the desired value and submit the request form.
