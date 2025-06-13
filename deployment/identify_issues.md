# 🚨 AWS デプロイスクリプトの問題点

## 発見された問題

### 1. **$500設定になっていない** 🚨
```bash
# 現在のスクリプト（間違い）
INITIAL_CAPITAL=10000
MAX_POSITION_SIZE_USD=350
SYMBOLS=BTCUSDT,ETHUSDT,ICPUSDT

# 正しい$500設定
INITIAL_CAPITAL=500
MAX_POSITION_SIZE_USD=25
SYMBOLS=ICPUSDT
```

### 2. **requirements.txtが不足** 🚨
```bash
# Dockerビルドがこける
ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'
```

### 3. **AMI IDが古い可能性** ⚠️
```bash
# 2024年のAMI IDを使用（最新化必要）
ami-0d52744d6551d851e  # 確認必要
```

### 4. **モデルファイルのアップロード未対応** ⚠️
```bash
# 手動でscpが必要
models/fast_nn_final.pth
models/fast_nn_scaler.pkl
```

### 5. **Gitリポジトリの同期問題** ⚠️
```bash
# ローカル変更がGitHubにpushされていない
# $500用の設定ファイルが含まれていない
```

### 6. **Docker Composeの環境変数問題** ⚠️
```bash
# 環境変数が正しく渡されない可能性
BYBIT_API_KEY=${BYBIT_API_KEY}
```

## 緊急修正が必要な項目

### 🔴 Critical (即座に修正)
1. $500用の設定に変更
2. requirements.txt作成
3. GitHubリポジトリ更新

### 🟡 Important (デプロイ前に修正)
4. AMI ID最新化
5. モデルファイル自動アップロード
6. 環境変数の確実な設定