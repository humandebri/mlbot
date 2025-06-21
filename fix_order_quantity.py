#!/usr/bin/env python3
"""
Fix order quantity precision for different symbols
"""

def get_fixed_execute_trade_method():
    """Return the fixed execute_trade method with proper quantity precision."""
    return '''
    async def execute_trade(self, symbol: str, prediction: float, confidence: float, features: Dict[str, float]):
        """Execute actual trade with risk management and proper quantity precision."""
        try:
            # Update balance before trading
            await self.update_balance()
            if not self.current_balance:
                logger.error("Cannot trade without balance information")
                return
            
            # Calculate position size
            position_size = self.current_balance * self.base_position_pct
            position_size = min(position_size, self.current_balance * 0.3)
            
            if position_size < self.min_order_size_usd:
                logger.warning(f"Position size ${position_size:.2f} below minimum")
                return
            
            # Get current price
            ticker = await self.bybit_client.get_ticker(symbol)
            if not ticker or "lastPrice" not in ticker:
                logger.error(f"Failed to get ticker for {symbol}")
                return
            
            current_price = float(ticker["lastPrice"])
            
            # Determine order side
            order_side = "buy" if prediction > 0.5 else "sell"
            
            # Risk management check
            if not self.risk_manager.can_trade(symbol=symbol, side=order_side, size=position_size):
                logger.warning(f"Risk manager blocked trade for {symbol}")
                discord_notifier.send_notification(
                    title="🛑 リスク管理ブロック",
                    description=f"{symbol} の取引がリスク管理によりブロックされました",
                    color="ff0000"
                )
                return
            
            # Calculate order parameters
            slippage = 0.001
            if order_side == "buy":
                order_price = current_price * (1 + slippage)
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.03
            else:
                order_price = current_price * (1 - slippage)
                stop_loss = current_price * 1.02
                take_profit = current_price * 0.97
            
            # Calculate order quantity with proper precision
            order_qty = position_size / current_price
            
            # Apply symbol-specific precision rules
            if symbol == "BTCUSDT":
                # BTC: 3 decimal places
                order_qty = round(order_qty, 3)
            elif symbol == "ETHUSDT":
                # ETH: 2 decimal places
                order_qty = round(order_qty, 2)
            elif symbol == "ICPUSDT":
                # ICP: 0 decimal places (whole numbers only)
                order_qty = int(order_qty)
                if order_qty < 1:
                    order_qty = 1  # Minimum 1 ICP
            else:
                # Default: 2 decimal places
                order_qty = round(order_qty, 2)
            
            # Ensure minimum order size
            min_order_values = {
                "BTCUSDT": 0.001,
                "ETHUSDT": 0.01,
                "ICPUSDT": 1
            }
            
            min_qty = min_order_values.get(symbol, 0.01)
            if order_qty < min_qty:
                logger.warning(f"Order quantity {order_qty} below minimum {min_qty} for {symbol}")
                return
            
            # Execute order
            logger.info(f"Executing {order_side} order for {symbol}: {order_qty} @ ${order_price:.2f}")
            
            order_result = await self.bybit_client.create_order(
                symbol=symbol,
                side=order_side,
                order_type="limit",
                qty=order_qty,
                price=order_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if order_result:
                order_id = order_result.get("orderId")
                position_id = f"pos_{order_id}"
                
                # Save to database
                save_position(
                    position_id=position_id,
                    symbol=symbol,
                    side=order_side,
                    entry_price=order_price,
                    quantity=order_qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "signal_confidence": confidence,
                        "ml_prediction": prediction,
                        "signal_time": datetime.now().isoformat()
                    }
                )
                
                save_trade(
                    trade_id=order_id,
                    position_id=position_id,
                    symbol=symbol,
                    side=order_side,
                    order_type="limit",
                    quantity=order_qty,
                    price=order_price,
                    metadata={
                        "signal_confidence": confidence,
                        "ml_prediction": prediction
                    }
                )
                
                # Send success notification
                discord_notifier.send_notification(
                    title="✅ 注文実行成功",
                    description=f"{symbol} の注文が正常に実行されました",
                    color="00ff00",
                    fields={
                        "Side": order_side.upper(),
                        "Quantity": f"{order_qty}",
                        "Price": f"${order_price:.2f}",
                        "Stop Loss": f"${stop_loss:.2f}",
                        "Take Profit": f"${take_profit:.2f}",
                        "Position Size": f"${position_size:.2f}",
                        "Confidence": f"{confidence*100:.1f}%",
                        "Order ID": order_id
                    }
                )
                
                logger.info(f"Order executed successfully: {order_id}")
                
            else:
                logger.error(f"Failed to execute order for {symbol}")
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            discord_notifier.send_notification(
                title="❌ 取引エラー",
                description=f"{symbol} の取引中にエラーが発生しました: {str(e)}",
                color="ff0000"
            )
'''

print("Order quantity fix created.")
print("\nKey changes:")
print("1. BTCUSDT: 3 decimal places (0.001 minimum)")
print("2. ETHUSDT: 2 decimal places (0.01 minimum)")
print("3. ICPUSDT: Whole numbers only (1 minimum)")
print("\nThis should fix the 'Qty invalid' error for ICPUSDT.")