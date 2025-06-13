# MLBot AWS デプロイ完全ガイド（$500版）

## 📋 事前準備チェックリスト

### 1. 必要なアカウント
- [ ] AWSアカウント（クレジットカード登録済み）
- [ ] Bybit本番アカウント（KYC完了）
- [ ] Discordアカウント（オプション）

### 2. ローカル環境の準備
```bash
# AWS CLIのインストール確認
aws --version

# インストールされていない場合
brew install awscli
```

### 3. AWS認証設定
```bash
aws configure
```
以下を入力:
- AWS Access Key ID: [IAMユーザーのアクセスキー]
- AWS Secret Access Key: [IAMユーザーのシークレットキー]
- Default region name: ap-northeast-1
- Default output format: json

※ アクセスキーの取得方法:
1. AWS Console → IAM → ユーザー
2. 「セキュリティ認証情報」タブ
3. 「アクセスキーを作成」

## 🚀 ステップ2: 自動デプロイ実行

### 1. デプロイディレクトリに移動
```bash
cd /Users/0xhude/Desktop/mlbot/deployment
```

### 2. IAMロール作成（初回のみ）
```bash
# EC2がSecrets Managerにアクセスするための権限設定
./iam_role_setup.sh
```

### 3. 自動デプロイスクリプト実行
```bash
./quick_aws_setup.sh
```

スクリプトが聞いてくること:
1. 準備できましたか？ → `y`
2. リージョン選択 → `1`（東京）
3. Bybit API Key → 本番APIキーを入力
4. Bybit API Secret → 本番APIシークレットを入力
5. Discord Webhook → URLを入力（またはEnterでスキップ）

### 4. 接続情報の確認
スクリプト完了後、`mlbot_connection_info.txt`が作成されます:
```
インスタンスID: i-xxxxxxxxx
パブリックIP: xx.xx.xx.xx
SSHキー: ~/.ssh/mlbot-key-xxxxx.pem
```

## 📦 ステップ3: アプリケーションセットアップ

### 1. モデルファイルのアップロード
```bash
# ローカルから実行
scp -i ~/.ssh/mlbot-key-*.pem models/fast_nn_final.pth ubuntu@[EC2のIP]:~/mlbot/models/
scp -i ~/.ssh/mlbot-key-*.pem models/fast_nn_scaler.pkl ubuntu@[EC2のIP]:~/mlbot/models/
```

### 2. サーバーに接続
```bash
ssh -i ~/.ssh/mlbot-key-*.pem ubuntu@[EC2のIP]
```

### 3. 環境設定（$500版）
```bash
cd mlbot

# 本番環境ファイルをコピー
cat > .env << 'EOF'
# Bybit API Configuration
BYBIT_API_KEY=will_be_loaded_from_secrets
BYBIT_API_SECRET=will_be_loaded_from_secrets
USE_TESTNET=false

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Discord Configuration
DISCORD_WEBHOOK=will_be_loaded_from_secrets

# System Configuration
LOG_LEVEL=INFO
ENVIRONMENT=production

# Trading Configuration - $500版
SYMBOLS=ICPUSDT
MIN_CONFIDENCE=0.65
MIN_EXPECTED_PNL=0.0015

# Risk Management - $500版
INITIAL_CAPITAL=500
MAX_POSITION_SIZE_USD=25
MAX_LEVERAGE=2
MAX_DAILY_LOSS_USD=25
MAX_DRAWDOWN_PCT=0.08
BASE_POSITION_SIZE_PCT=0.05

# Execution Configuration
USE_POST_ONLY=true
PRICE_BUFFER_PCT=0.0003
MAX_ORDER_AGE_SECONDS=180
AGGRESSIVE_FILL_TIMEOUT=20
EOF
```

### 4. セットアップスクリプト実行
```bash
# 初期セットアップ
bash ../setup_mlbot.sh

# シークレット読み込み
source load_secrets.sh
```

## 🐳 ステップ4: Docker起動

### 1. Dockerイメージのビルド
```bash
docker-compose build
```

### 2. サービス起動
```bash
# バックグラウンドで起動
docker-compose up -d

# ログを見ながら起動（推奨）
docker-compose up
```

### 3. 動作確認
別のターミナルで:
```bash
# ヘルスチェック
curl http://localhost:8080/health

# ログ確認
docker-compose logs -f mlbot
```

## 🔍 ステップ5: 監視と確認

### 1. Discord通知確認
Discord通知が来れば正常動作:
- 「🚀 MLBot起動」メッセージ
- デバイス情報
- 初期資本: $500

### 2. 初回取引の監視
```bash
# リアルタイムログ
docker-compose logs -f mlbot | grep -E "(Order|Position|Signal)"
```

### 3. トラブルシューティング

#### ケース1: Dockerが起動しない
```bash
# エラーログ確認
docker-compose logs

# 再起動
docker-compose down
docker-compose up
```

#### ケース2: API接続エラー
```bash
# シークレット再読み込み
source load_secrets.sh
env | grep BYBIT  # 環境変数確認

# 手動で環境変数設定
export BYBIT_API_KEY="your_key"
export BYBIT_API_SECRET="your_secret"
```

#### ケース3: モデルファイルエラー
```bash
# ファイル確認
ls -la models/

# 権限修正
chmod 644 models/*
```

## 📊 ステップ6: 運用開始

### 1. 初期動作確認（1-2時間）
- 取引シグナルが生成されるか
- 注文が正しく配置されるか
- Discord通知が届くか

### 2. 日次チェック
```bash
# SSHで接続
ssh -i ~/.ssh/mlbot-key-*.pem ubuntu@[EC2のIP]

# ステータス確認
cd mlbot
docker-compose ps
docker-compose logs --tail 100

# メトリクス確認
curl http://localhost:8080/metrics
```

### 3. 停止方法
```bash
# 一時停止
docker-compose stop

# 完全停止
docker-compose down

# 再開
docker-compose up -d
```

## 💰 コスト管理

### EC2の停止（取引しない時）
```bash
# ローカルから
aws ec2 stop-instances --instance-ids [インスタンスID] --region ap-northeast-1

# 再開
aws ec2 start-instances --instance-ids [インスタンスID] --region ap-northeast-1
```

### 月額コスト
- EC2 t3.medium: ~$30
- EBS 100GB: ~$8
- データ転送: ~$2
- **合計: 約$40（5,600円）**

## 🎯 次のステップ

1. **1週間**: 最小ポジション（$12-15）で動作確認
2. **2週間目**: ポジションサイズを$25に増加
3. **1ヶ月後**: 結果を分析、設定調整
4. **$1,000到達時**: 通常設定に移行

## ⚠️ 重要な注意事項

1. **必ず$500の少額から開始**
2. **最初の1週間は様子見**
3. **感情的な介入をしない**
4. **システムを信頼する**
5. **日次で結果を記録**

---

サポートが必要な場合:
- GitHub Issues: https://github.com/humandebri/mlbot/issues
- エラーログと共に報告してください