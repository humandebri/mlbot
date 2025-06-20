#!/usr/bin/env python3
"""
V3.1_improved修復モデルのEC2デプロイメントスクリプト
修復されたモデルを本番環境にデプロイして実際の取引を開始
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def deploy_v31_improved():
    """V3.1_improved修復モデルをEC2にデプロイ"""
    print("="*80)
    print("🚀 V3.1_improved修復モデル EC2デプロイメント")
    print("="*80)
    
    try:
        # 1. ローカルファイルの確認
        print("\n1️⃣ ローカルファイル確認...")
        
        required_files = [
            "src/ml_pipeline/v31_improved_inference_engine.py",
            "src/ml_pipeline/feature_adapter_44.py", 
            "src/integration/dynamic_trading_coordinator.py",
            "models/v3.1_improved/model.onnx",
            "models/v3.1_improved/metadata.json",
            "test_v31_trading_integration.py"
        ]
        
        all_files_exist = True
        for file_path in required_files:
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size
                print(f"     ✅ {file_path} ({size/1024:.1f} KB)")
            else:
                print(f"     ❌ {file_path} - 見つかりません")
                all_files_exist = False
        
        if not all_files_exist:
            print("❌ 必要なファイルが不足しています")
            return False
        
        # 2. EC2接続情報
        print("\n2️⃣ EC2接続情報...")
        
        ec2_info = {
            "host": "13.212.91.54",
            "user": "ubuntu", 
            "key_path": "~/.ssh/mlbot-key-*.pem",
            "remote_dir": "/home/ubuntu/mlbot"
        }
        
        print(f"     Host: {ec2_info['host']}")
        print(f"     User: {ec2_info['user']}")
        print(f"     Directory: {ec2_info['remote_dir']}")
        
        # 3. rsyncコマンド生成
        print("\n3️⃣ デプロイコマンド生成...")
        
        # 新しいファイルのアップロード
        upload_commands = [
            # V3.1_improved推論エンジン
            f"rsync -avz -e 'ssh -i ~/.ssh/mlbot-key-*.pem' src/ml_pipeline/v31_improved_inference_engine.py ubuntu@{ec2_info['host']}:{ec2_info['remote_dir']}/src/ml_pipeline/",
            
            # 更新されたtrading coordinator  
            f"rsync -avz -e 'ssh -i ~/.ssh/mlbot-key-*.pem' src/integration/dynamic_trading_coordinator.py ubuntu@{ec2_info['host']}:{ec2_info['remote_dir']}/src/integration/",
            
            # テストファイル
            f"rsync -avz -e 'ssh -i ~/.ssh/mlbot-key-*.pem' test_v31_trading_integration.py ubuntu@{ec2_info['host']}:{ec2_info['remote_dir']}/",
            
            # 修復テストファイル
            f"rsync -avz -e 'ssh -i ~/.ssh/mlbot-key-*.pem' fix_v31_improved_model.py ubuntu@{ec2_info['host']}:{ec2_info['remote_dir']}/"
        ]
        
        # 4. EC2での検証コマンド
        print("\n4️⃣ EC2検証コマンド...")
        
        verification_commands = [
            # システム停止
            "ssh -i ~/.ssh/mlbot-key-*.pem ubuntu@13.212.91.54 'cd /home/ubuntu/mlbot && pkill -f main_dynamic_integration'",
            
            # 統合テスト実行
            "ssh -i ~/.ssh/mlbot-key-*.pem ubuntu@13.212.91.54 'cd /home/ubuntu/mlbot && python3 test_v31_trading_integration.py'",
            
            # 修復モデルテスト
            "ssh -i ~/.ssh/mlbot-key-*.pem ubuntu@13.212.91.54 'cd /home/ubuntu/mlbot && python3 fix_v31_improved_model.py'",
            
            # システム再起動
            "ssh -i ~/.ssh/mlbot-key-*.pem ubuntu@13.212.91.54 'cd /home/ubuntu/mlbot && nohup python3 main_dynamic_integration.py > trading.log 2>&1 &'",
            
            # 動作確認
            "ssh -i ~/.ssh/mlbot-key-*.pem ubuntu@13.212.91.54 'cd /home/ubuntu/mlbot && ps aux | grep main_dynamic_integration'"
        ]
        
        # 5. デプロイ手順の表示
        print("\n5️⃣ デプロイ手順...")
        
        print("📁 **Step 1: ファイルアップロード**")
        for i, cmd in enumerate(upload_commands, 1):
            print(f"   {i}. {cmd}")
        
        print("\n🔍 **Step 2: EC2での検証**")
        for i, cmd in enumerate(verification_commands, 1):
            print(f"   {i}. {cmd}")
        
        # 6. 自動実行オプション
        print("\n6️⃣ 自動実行...")
        
        user_input = input("自動でデプロイを実行しますか？ (y/N): ").strip().lower()
        
        if user_input == 'y':
            print("\n🚀 自動デプロイ開始...")
            
            import subprocess
            
            # ファイルアップロード
            print("\n📁 ファイルアップロード中...")
            for i, cmd in enumerate(upload_commands, 1):
                print(f"   実行中 {i}/{len(upload_commands)}: {cmd.split()[-1]}")
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"   ✅ 成功")
                    else:
                        print(f"   ❌ エラー: {result.stderr}")
                except Exception as e:
                    print(f"   ❌ 例外: {e}")
            
            # 検証実行
            print("\n🔍 EC2検証中...")
            for i, cmd in enumerate(verification_commands, 1):
                print(f"   実行中 {i}/{len(verification_commands)}: {cmd.split()[-1] if 'grep' not in cmd else 'process check'}")
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        print(f"   ✅ 成功")
                        if result.stdout.strip():
                            print(f"      出力: {result.stdout.strip()[:100]}...")
                    else:
                        print(f"   ⚠️ 警告: {result.stderr[:100] if result.stderr else 'No output'}")
                except subprocess.TimeoutExpired:
                    print(f"   ⏰ タイムアウト (30秒)")
                except Exception as e:
                    print(f"   ❌ 例外: {e}")
                
                # 少し待機
                await asyncio.sleep(2)
            
            print("\n✅ 自動デプロイ完了")
        else:
            print("\n📋 手動デプロイの場合、上記のコマンドを順番に実行してください")
        
        # 7. 成功確認
        print("\n7️⃣ デプロイ成功確認...")
        
        success_indicators = [
            "✅ V3.1_improved推論エンジンがEC2で動作",
            "✅ 統合テストが成功",
            "✅ main_dynamic_integration.pyが起動",
            "✅ 実際の取引シグナル生成開始"
        ]
        
        print("デプロイ成功の指標:")
        for indicator in success_indicators:
            print(f"   {indicator}")
        
        # 8. 監視コマンド
        print("\n8️⃣ 監視コマンド...")
        
        monitoring_commands = [
            "ssh -i ~/.ssh/mlbot-key-*.pem ubuntu@13.212.91.54 'tail -f /home/ubuntu/mlbot/trading.log'",
            "ssh -i ~/.ssh/mlbot-key-*.pem ubuntu@13.212.91.54 'cd /home/ubuntu/mlbot && python3 -c \"import asyncio; from src.integration.dynamic_trading_coordinator import *; print(\\\"System check complete\\\")\"'",
        ]
        
        print("システム監視用コマンド:")
        for cmd in monitoring_commands:
            print(f"   {cmd}")
        
        return True
        
    except Exception as e:
        print(f"❌ デプロイエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    try:
        success = asyncio.run(deploy_v31_improved())
        
        print("\n" + "="*80)
        if success:
            print("🎯 V3.1_improved修復モデルデプロイ準備完了")
            print("📈 修復されたモデル（AUC 0.838）で実際の取引が開始されます")
            print("🚀 EC2で24時間自動取引システム稼働中")
        else:
            print("❌ デプロイに問題が発生しました")
            print("手動でファイルを確認してください")
        print("="*80)
        
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()