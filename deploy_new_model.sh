#!/bin/bash

echo "🚀 新モデル（v4）のデプロイメント計画"
echo "===================================="

# カラー定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ステップ1: データ準備
echo -e "\n${GREEN}Step 1: データ準備${NC}"
echo "1. DuckDBから6ヶ月分のデータを抽出"
echo "2. クラスバランシングの実施（SMOTE + アンダーサンプリング）"
echo "3. 52次元の特徴量（既存44 + 新規8）"
echo ""
echo "実行コマンド:"
echo -e "${YELLOW}python prepare_balanced_dataset.py${NC}"
echo ""

# ステップ2: モデル訓練
echo -e "\n${GREEN}Step 2: アンサンブルモデルの訓練${NC}"
echo "モデル構成:"
echo "  - LSTM: 時系列パターン学習"
echo "  - Transformer: Attention機構"
echo "  - XGBoost: 勾配ブースティング"
echo "  - Neural Network: 深層学習"
echo ""
echo "特徴:"
echo "  - Focal Loss使用（クラス不均衡対策）"
echo "  - リアルタイム予測バイアス監視"
echo "  - アンサンブル重み最適化"
echo ""
echo "実行コマンド:"
echo -e "${YELLOW}python train_ensemble_model.py${NC}"
echo ""

# ステップ3: 検証
echo -e "\n${GREEN}Step 3: モデル検証${NC}"
echo "検証項目:"
echo "  ✓ Buy/Sell比率: 40-60%の範囲内"
echo "  ✓ AUC: 0.85以上"
echo "  ✓ 予測分布: 0.1-0.9の範囲"
echo "  ✓ 時間帯別の安定性"
echo ""

# ステップ4: ONNX変換
echo -e "\n${GREEN}Step 4: ONNX形式への変換${NC}"
cat > convert_to_onnx.py << 'EOF'
import tensorflow as tf
import tf2onnx
import onnx

# モデル読み込み
model = tf.keras.models.load_model('models/v4_ensemble/lstm_model.h5')

# ONNX変換
spec = (tf.TensorSpec((None, 52), tf.float32, name="float_input"),)
output_path = "models/v4_ensemble/model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, spec, output_path=output_path)
print(f"✅ Model converted to {output_path}")
EOF

echo "実行コマンド:"
echo -e "${YELLOW}python convert_to_onnx.py${NC}"
echo ""

# ステップ5: A/Bテスト設定
echo -e "\n${GREEN}Step 5: A/Bテストデプロイ${NC}"
cat > ab_test_config.json << 'EOF'
{
  "test_name": "v3.1_vs_v4_ensemble",
  "models": {
    "control": {
      "path": "models/v3.1_improved/model.onnx",
      "allocation": 0.7
    },
    "treatment": {
      "path": "models/v4_ensemble/model.onnx",
      "allocation": 0.3
    }
  },
  "metrics": [
    "prediction_balance",
    "signal_quality",
    "profitability"
  ],
  "duration": "72h"
}
EOF

# ステップ6: EC2デプロイ
echo -e "\n${GREEN}Step 6: EC2へのデプロイ${NC}"
echo "デプロイ手順:"
echo "1. モデルファイルをEC2に転送"
echo "   scp -r models/v4_ensemble/ ubuntu@EC2_IP:/home/ubuntu/mlbot/models/"
echo ""
echo "2. A/Bテスト版ボットを起動"
echo "   python simple_improved_bot_ab_test.py --config ab_test_config.json"
echo ""
echo "3. メトリクス監視"
echo "   - Discord通知でリアルタイム監視"
echo "   - Buy/Sell比率の追跡"
echo "   - 収益性の比較"
echo ""

# タイムライン
echo -e "\n${GREEN}📅 タイムライン${NC}"
echo "Day 1-2: データ準備とクラスバランシング"
echo "Day 3-4: モデル訓練と検証"
echo "Day 5:   ONNX変換とA/Bテスト準備"
echo "Day 6-8: 本番環境でのA/Bテスト"
echo "Day 9:   結果分析と完全移行判断"
echo ""

# 注意事項
echo -e "\n${RED}⚠️  重要な注意事項${NC}"
echo "1. 既存のv3.1モデルは削除せず、バックアップとして保持"
echo "2. A/Bテスト中は両モデルの性能を常時監視"
echo "3. 問題発生時は即座にv3.1に戻せる体制を維持"
echo "4. 最低72時間のテスト期間を確保"
echo ""

echo -e "${GREEN}準備ができたら、上記のコマンドを順番に実行してください。${NC}"