#!/usr/bin/env python3
"""
統合テスト: 特徴量生成が正しく動作することを確認
要件: 全シンボルで特徴量カウント > 0
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.config import settings
from src.common.logging import get_logger
from src.integration.simple_service_manager import SimpleServiceManager

logger = get_logger(__name__)


class FeatureGenerationTest:
    """特徴量生成の統合テスト"""
    
    def __init__(self):
        self.service_manager = SimpleServiceManager()
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
    
    async def run_test(self, duration_seconds: int = 30):
        """
        テストを実行
        Args:
            duration_seconds: テスト実行時間（秒）
        """
        logger.info(f"Starting feature generation test for {duration_seconds} seconds")
        
        try:
            # サービスを開始
            await self.service_manager.start_all()
            logger.info("All services started")
            
            # 初期データ蓄積のため待機
            logger.info("Waiting for initial data accumulation...")
            await asyncio.sleep(10)
            
            # 定期的に特徴量をチェック
            start_time = asyncio.get_event_loop().time()
            check_count = 0
            
            while asyncio.get_event_loop().time() - start_time < duration_seconds:
                check_count += 1
                logger.info(f"\n=== Check #{check_count} ===")
                
                # サービスステータスを取得
                status = self.service_manager.get_service_status()
                
                # 各チェック項目を確認
                await self._check_service_status(status)
                await self._check_feature_counts(status)
                await self._check_data_flow()
                
                # 5秒待機
                await asyncio.sleep(5)
            
            # 最終結果を表示
            self._print_results()
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}", exc_info=True)
            self.test_results["failed"] += 1
            self.test_results["errors"].append(str(e))
        finally:
            # サービスを停止
            await self.service_manager.stop_all()
            logger.info("All services stopped")
    
    async def _check_service_status(self, status: dict):
        """サービスの稼働状態をチェック"""
        logger.info("Checking service status...")
        
        # Ingestorチェック
        if status["ingestor"]["running"]:
            logger.info("✅ Ingestor: Running")
            self.test_results["passed"] += 1
        else:
            logger.error("❌ Ingestor: Not running")
            self.test_results["failed"] += 1
            self.test_results["errors"].append("Ingestor not running")
        
        # FeatureHubチェック
        if status["feature_hub"]["running"]:
            logger.info("✅ FeatureHub: Running")
            self.test_results["passed"] += 1
        else:
            logger.error("❌ FeatureHub: Not running")
            self.test_results["failed"] += 1
            self.test_results["errors"].append("FeatureHub not running")
        
        # バックグラウンドタスクチェック
        bg_tasks = status["background_tasks"]
        if bg_tasks["running"] >= 4:  # FeatureHubの4つのタスク
            logger.info(f"✅ Background tasks: {bg_tasks['running']}/{bg_tasks['total']} running")
            self.test_results["passed"] += 1
        else:
            logger.error(f"❌ Background tasks: Only {bg_tasks['running']}/{bg_tasks['total']} running")
            self.test_results["failed"] += 1
            self.test_results["errors"].append("Not all background tasks running")
    
    async def _check_feature_counts(self, status: dict):
        """特徴量カウントをチェック（最重要）"""
        logger.info("Checking feature counts...")
        
        feature_counts = status["feature_hub"]["feature_counts"]
        all_symbols_have_features = True
        
        for symbol in settings.bybit.symbols:
            count = feature_counts.get(symbol, 0)
            if count > 0:
                logger.info(f"✅ {symbol}: {count} features")
                self.test_results["passed"] += 1
            else:
                logger.error(f"❌ {symbol}: 0 features")
                self.test_results["failed"] += 1
                self.test_results["errors"].append(f"{symbol} has 0 features")
                all_symbols_have_features = False
        
        if all_symbols_have_features:
            logger.info("🎉 All symbols have features generated!")
        else:
            logger.error("⚠️ Some symbols have no features")
    
    async def _check_data_flow(self):
        """データフローをチェック"""
        logger.info("Checking data flow...")
        
        # Ingestorのメッセージレートを確認
        if hasattr(self.service_manager.ingestor, 'message_rate'):
            rate = self.service_manager.ingestor.message_rate
            if rate > 0:
                logger.info(f"✅ Ingestor message rate: {rate:.1f} msg/s")
                self.test_results["passed"] += 1
            else:
                logger.error("❌ Ingestor message rate: 0 msg/s")
                self.test_results["failed"] += 1
                self.test_results["errors"].append("No data flow from Ingestor")
        
        # FeatureHubの処理を確認
        if self.service_manager.feature_hub:
            # PriceFeatureEngineの存在確認
            if hasattr(self.service_manager.feature_hub, 'price_engine'):
                if self.service_manager.feature_hub.price_engine is not None:
                    logger.info("✅ PriceFeatureEngine: Initialized")
                    self.test_results["passed"] += 1
                    
                    # latest_features属性の確認
                    if hasattr(self.service_manager.feature_hub.price_engine, 'latest_features'):
                        logger.info("✅ PriceFeatureEngine.latest_features: Exists")
                        self.test_results["passed"] += 1
                    else:
                        logger.error("❌ PriceFeatureEngine.latest_features: Missing")
                        self.test_results["failed"] += 1
                        self.test_results["errors"].append("latest_features attribute missing")
                else:
                    logger.error("❌ PriceFeatureEngine: Not initialized")
                    self.test_results["failed"] += 1
                    self.test_results["errors"].append("PriceFeatureEngine not initialized")
    
    def _print_results(self):
        """テスト結果を表示"""
        total_tests = self.test_results["passed"] + self.test_results["failed"]
        
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {self.test_results['passed']} ✅")
        print(f"Failed: {self.test_results['failed']} ❌")
        print(f"Success rate: {self.test_results['passed']/total_tests*100:.1f}%")
        
        if self.test_results["errors"]:
            print("\nErrors:")
            for error in self.test_results["errors"]:
                print(f"  - {error}")
        
        print("\n" + "="*60)
        
        # 成功判定
        if self.test_results["failed"] == 0:
            print("🎉 ALL TESTS PASSED! System is ready for deployment.")
        else:
            print("❌ TESTS FAILED! Please fix the issues before deployment.")


async def main():
    """メイン実行関数"""
    test = FeatureGenerationTest()
    await test.run_test(duration_seconds=30)


if __name__ == "__main__":
    print("Feature Generation Integration Test")
    print("="*60)
    asyncio.run(main())