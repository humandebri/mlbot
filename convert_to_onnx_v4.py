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
