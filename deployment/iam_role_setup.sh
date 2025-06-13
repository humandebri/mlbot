#!/bin/bash
# IAMロール作成スクリプト（Secrets Manager アクセス用）

REGION=${1:-"ap-northeast-1"}

echo "Creating IAM role for EC2 to access Secrets Manager..."

# Trust policy
cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
    --role-name mlbot-ec2-role \
    --assume-role-policy-document file://trust-policy.json \
    --region $REGION

# Attach policy for Secrets Manager
cat > secrets-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue",
                "secretsmanager:DescribeSecret"
            ],
            "Resource": [
                "arn:aws:secretsmanager:${REGION}:*:secret:mlbot/*"
            ]
        }
    ]
}
EOF

aws iam put-role-policy \
    --role-name mlbot-ec2-role \
    --policy-name mlbot-secrets-policy \
    --policy-document file://secrets-policy.json

# Create instance profile
aws iam create-instance-profile \
    --instance-profile-name mlbot-instance-profile

# Add role to instance profile
aws iam add-role-to-instance-profile \
    --instance-profile-name mlbot-instance-profile \
    --role-name mlbot-ec2-role

# Cleanup
rm trust-policy.json secrets-policy.json

echo "✅ IAM role created successfully!"
echo "Instance profile: mlbot-instance-profile"