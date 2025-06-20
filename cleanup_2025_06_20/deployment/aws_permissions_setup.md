# AWS IAMユーザー権限設定ガイド

## 🔐 必要な権限の設定

### ステップ1: IAMユーザーの権限確認

作成したアクセスキーに以下の権限が必要です：

#### 最小限の権限（推奨）
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:RunInstances",
                "ec2:TerminateInstances",
                "ec2:DescribeInstances",
                "ec2:DescribeImages",
                "ec2:DescribeKeyPairs",
                "ec2:DescribeSecurityGroups",
                "ec2:DescribeSubnets",
                "ec2:DescribeVpcs",
                "ec2:CreateKeyPair",
                "ec2:CreateSecurityGroup",
                "ec2:AuthorizeSecurityGroupIngress",
                "ec2:CreateTags",
                "ec2:StopInstances",
                "ec2:StartInstances"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:CreateSecret",
                "secretsmanager:GetSecretValue",
                "secretsmanager:UpdateSecret",
                "secretsmanager:DeleteSecret"
            ],
            "Resource": "arn:aws:secretsmanager:*:*:secret:mlbot/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:CreateRole",
                "iam:CreateInstanceProfile",
                "iam:AddRoleToInstanceProfile",
                "iam:PutRolePolicy",
                "iam:PassRole"
            ],
            "Resource": [
                "arn:aws:iam::*:role/mlbot-*",
                "arn:aws:iam::*:instance-profile/mlbot-*"
            ]
        }
    ]
}
```

## 🚀 クイック設定（管理者権限使用）

### 方法1: AWS管理ポリシーを使用（簡単）

AWS Consoleで以下のポリシーを追加：

1. **AmazonEC2FullAccess**
2. **SecretsManagerReadWrite** 
3. **IAMFullAccess**（一時的）

⚠️ 注意: 本番環境では最小権限の原則に従ってください

### 方法2: カスタムポリシー作成（推奨）

1. AWS Console → IAM → ポリシー
2. 「ポリシーの作成」をクリック
3. JSONタブで上記の最小権限ポリシーを貼り付け
4. ポリシー名: `MLBotDeploymentPolicy`
5. 作成したポリシーをユーザーにアタッチ

## 📋 手順詳細

### 1. AWS Consoleにログイン
- https://aws.amazon.com/console/

### 2. IAMサービスに移動
- サービス → IAM

### 3. ユーザーを選択
- 左メニュー「ユーザー」
- 作成したユーザーをクリック

### 4. 権限を追加
- 「許可を追加」ボタン
- 「ポリシーを直接アタッチ」

### 5. 必要なポリシーを選択

#### オプション A: 管理ポリシー（簡単）
```
✅ AmazonEC2FullAccess
✅ SecretsManagerReadWrite
✅ IAMFullAccess
```

#### オプション B: カスタムポリシー（セキュア）
- 上記のJSONポリシーを使用

### 6. 権限の確認
```bash
# 権限テスト
aws sts get-caller-identity
aws ec2 describe-regions
```

## 🔧 トラブルシューティング

### エラー1: "User is not authorized"
```bash
# 解決: EC2権限を追加
aws iam attach-user-policy \
    --user-name YOUR_USERNAME \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess
```

### エラー2: "Cannot create secret"
```bash
# 解決: Secrets Manager権限を追加
aws iam attach-user-policy \
    --user-name YOUR_USERNAME \
    --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
```

### エラー3: "Cannot create IAM role"
```bash
# 解決: IAM権限を追加
aws iam attach-user-policy \
    --user-name YOUR_USERNAME \
    --policy-arn arn:aws:iam::aws:policy/IAMFullAccess
```

## ⚡ 最速セットアップ（5分）

管理者権限がある場合：

```bash
# 1. ユーザー名を環境変数に設定
export AWS_USERNAME="your-username"

# 2. 必要な権限を一括付与
aws iam attach-user-policy \
    --user-name $AWS_USERNAME \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess

aws iam attach-user-policy \
    --user-name $AWS_USERNAME \
    --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite

aws iam attach-user-policy \
    --user-name $AWS_USERNAME \
    --policy-arn arn:aws:iam::aws:policy/IAMFullAccess

# 3. 確認
aws sts get-caller-identity
```

## 🛡️ セキュリティ考慮事項

### 本番環境での推奨事項

1. **最小権限の原則**
   - 必要最小限の権限のみ付与
   - 定期的な権限レビュー

2. **一時的権限**
   - IAMFullAccessは削除可能
   - デプロイ後に権限を最小化

3. **アクセスキーの管理**
   - 定期的なローテーション
   - 不要になったら削除

### デプロイ後のクリーンアップ

```bash
# IAMFullAccessを削除（デプロイ完了後）
aws iam detach-user-policy \
    --user-name $AWS_USERNAME \
    --policy-arn arn:aws:iam::aws:policy/IAMFullAccess
```

## ✅ 権限設定完了チェックリスト

- [ ] IAMユーザーにEC2権限を付与
- [ ] Secrets Manager権限を付与  
- [ ] IAM管理権限を付与（一時的）
- [ ] `aws sts get-caller-identity`で確認
- [ ] `aws ec2 describe-regions`で確認
- [ ] デプロイスクリプトの実行準備完了

---

権限設定が完了したら、デプロイスクリプトを実行できます：

```bash
cd /Users/0xhude/Desktop/mlbot/deployment
./quick_aws_setup.sh
```