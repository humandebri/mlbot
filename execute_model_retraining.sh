#!/bin/bash

echo "🚀 モデル再訓練実行スクリプト"
echo "================================"
echo ""

# カラー定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# 作業ディレクトリ
WORK_DIR="/Users/0xhude/Desktop/mlbot"
cd $WORK_DIR

# ログディレクトリ作成
mkdir -p logs/retraining
LOG_FILE="logs/retraining/retraining_$(date +%Y%m%d_%H%M%S).log"

# ロギング関数
log() {
    echo -e "$1" | tee -a $LOG_FILE
}

# エラーハンドリング
set -e
trap 'log "${RED}❌ エラーが発生しました。ログを確認してください: $LOG_FILE${NC}"' ERR

# ステップ1: 環境確認
log "${BLUE}=== Step 1: 環境確認 ===${NC}"
log "Pythonバージョン:"
python3 --version | tee -a $LOG_FILE

log "\n必要なライブラリ確認:"
python3 -c "import pandas, numpy, duckdb, tensorflow, xgboost, imblearn; print('✅ すべてのライブラリが利用可能')" 2>&1 | tee -a $LOG_FILE || {
    log "${RED}必要なライブラリが不足しています。requirements.txtを確認してください。${NC}"
    exit 1
}

# ステップ2: データ統合
log "\n${BLUE}=== Step 2: 履歴データの統合 ===${NC}"
log "既存の2-4年分のデータを統合中..."

python3 merge_historical_data.py 2>&1 | tee -a $LOG_FILE

if [ $? -eq 0 ]; then
    log "${GREEN}✅ データ統合完了${NC}"
else
    log "${RED}❌ データ統合に失敗しました${NC}"
    exit 1
fi

# ステップ3: バランスデータセット作成
log "\n${BLUE}=== Step 3: バランスデータセット作成 ===${NC}"
log "52次元の特徴量でバランスの取れたデータセットを作成中..."

python3 prepare_balanced_dataset_v2.py 2>&1 | tee -a $LOG_FILE

if [ $? -eq 0 ]; then
    log "${GREEN}✅ データセット作成完了${NC}"
    
    # データセットの統計を表示
    python3 -c "
import numpy as np
import json

# データセット読み込み
data = np.load('data/balanced_dataset_v4_full.npz')

# メタデータ読み込み
with open('data/balanced_dataset_v4_full_metadata.json', 'r') as f:
    metadata = json.load(f)

print('')
print('📊 データセット統計:')
print(f'  訓練: {metadata[\"train_samples\"]:,}件 (Buy: {metadata[\"train_buy_ratio\"]*100:.1f}%)')
print(f'  検証: {metadata[\"val_samples\"]:,}件 (Buy: {metadata[\"val_buy_ratio\"]*100:.1f}%)')
print(f'  テスト: {metadata[\"test_samples\"]:,}件 (Buy: {metadata[\"test_buy_ratio\"]*100:.1f}%)')
print(f'  特徴量: {metadata[\"n_features\"]}次元')
    " 2>&1 | tee -a $LOG_FILE
else
    log "${RED}❌ データセット作成に失敗しました${NC}"
    exit 1
fi

# ステップ4: モデル訓練
log "\n${BLUE}=== Step 4: アンサンブルモデル訓練 ===${NC}"
log "4つのモデル（LSTM, Transformer, XGBoost, NN）を訓練中..."
log "${YELLOW}⚠️  この処理には数時間かかる場合があります${NC}"

# GPUの確認
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'🎮 GPU利用可能: {len(gpus)}個のGPU')
else:
    print('💻 CPU mode')
" 2>&1 | tee -a $LOG_FILE

# モデル訓練実行
python3 train_ensemble_model.py 2>&1 | tee -a $LOG_FILE

if [ $? -eq 0 ]; then
    log "${GREEN}✅ モデル訓練完了${NC}"
    
    # 訓練結果の確認
    if [ -f "models/v4_ensemble/metadata.json" ]; then
        log "\n📊 訓練結果:"
        python3 -c "
import json
with open('models/v4_ensemble/metadata.json', 'r') as f:
    metadata = json.load(f)
print(f'  作成日時: {metadata[\"created_at\"]}')
print(f'  モデル: {metadata[\"model_names\"]}')
print(f'  アンサンブル重み: {metadata[\"ensemble_weights\"]}')
        " 2>&1 | tee -a $LOG_FILE
    fi
else
    log "${RED}❌ モデル訓練に失敗しました${NC}"
    exit 1
fi

# ステップ5: ONNX変換の準備
log "\n${BLUE}=== Step 5: ONNX変換準備 ===${NC}"

# ONNX変換スクリプトを作成
cat > convert_to_onnx_v4.py << 'EOF'
#!/usr/bin/env python3
"""
訓練済みモデルをONNX形式に変換
"""

import tensorflow as tf
import tf2onnx
import onnx
import os
import json

def convert_keras_to_onnx(model_path, output_path, input_shape):
    """KerasモデルをONNXに変換"""
    print(f"Converting {model_path} to ONNX...")
    
    # モデル読み込み
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # ONNX変換
    spec = (tf.TensorSpec((None, input_shape), tf.float32, name="float_input"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model, 
        input_signature=spec,
        output_path=output_path,
        opset=13
    )
    
    print(f"✅ Converted to {output_path}")
    
    # 検証
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model validation passed")
    
    return output_path

def main():
    # 52次元の入力
    input_shape = 52
    
    # 各モデルを変換
    models_to_convert = [
        ('lstm', 'models/v4_ensemble/lstm_model.h5'),
        ('transformer', 'models/v4_ensemble/transformer_model.h5'),
        ('neural_net', 'models/v4_ensemble/neural_net_model.h5')
    ]
    
    converted_models = []
    
    for name, keras_path in models_to_convert:
        if os.path.exists(keras_path):
            onnx_path = keras_path.replace('.h5', '.onnx')
            try:
                convert_keras_to_onnx(keras_path, onnx_path, input_shape)
                converted_models.append(name)
            except Exception as e:
                print(f"❌ Failed to convert {name}: {e}")
    
    # メタデータ更新
    metadata_path = 'models/v4_ensemble/metadata.json'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['onnx_models'] = converted_models
        metadata['input_shape'] = input_shape
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Conversion complete! Converted {len(converted_models)} models to ONNX")

if __name__ == "__main__":
    main()
EOF

log "ONNX変換スクリプトを作成しました"

# 最終サマリー
log "\n${GREEN}=== 訓練完了サマリー ===${NC}"
log "✅ データ統合: 2-4年分の履歴データ（660万レコード）"
log "✅ 特徴量: 52次元（既存44 + 新規8）"
log "✅ バランシング: Buy/Sell比率を40-60%に調整"
log "✅ モデル: LSTM, Transformer, XGBoost, Neural Networkのアンサンブル"
log ""
log "📁 出力ファイル:"
log "  - data/balanced_dataset_v4_full.npz"
log "  - models/v4_ensemble/"
log ""
log "${YELLOW}次のステップ:${NC}"
log "1. python3 convert_to_onnx_v4.py でONNX変換"
log "2. A/Bテストの実施"
log "3. 本番環境へのデプロイ"
log ""
log "詳細ログ: $LOG_FILE"
log "${GREEN}✅ モデル再訓練プロセス完了！${NC}"