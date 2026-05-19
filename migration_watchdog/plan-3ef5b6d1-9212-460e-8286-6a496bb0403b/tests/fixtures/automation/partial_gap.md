# GCP to AWS Migration Guide

This guide covers migrating your workloads from Google Cloud Platform to Amazon Web Services.

## Requesting Service Quota Increases

During migration, you may need to request quota increases for AWS services.
Navigate to the AWS Service Quotas console and search for the service you need.
Select the quota you want to increase and click "Request quota increase".
Fill in the desired value and submit the request.

## Generated Script

The following script checks your current quota limits:

```bash
#!/bin/bash
# check_quotas.sh - Check current service quota limits
aws service-quotas list-service-quotas --service-code ec2
```

Note: This script only checks quotas but does not request increases.
You must manually submit quota increase requests through the console.
