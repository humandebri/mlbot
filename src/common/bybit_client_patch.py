    async def place_order(self, symbol: str, side: str, qty: float, price: float = None, order_type: str = "Limit", time_in_force: str = "GTC") -> Dict[str, Any]:
        """Place an order on Bybit."""
        try:
            async with self.request_semaphore:
                await self._wait_for_rate_limit()
                
                url = urljoin(self.base_url, "/v5/order/create")
                
                order_data = {
                    "category": "linear",
                    "symbol": symbol,
                    "side": side.title(),  # Buy or Sell
                    "orderType": order_type,
                    "qty": str(qty),
                    "timeInForce": time_in_force
                }
                
                if price is not None:
                    order_data["price"] = str(price)
                
                headers = self._get_authenticated_headers("POST", "/v5/order/create", order_data)
                
                async with self.session.post(url, headers=headers, json=order_data) as response:
                    data = await response.json()
                    
                    if response.status == 200 and data.get("retCode") == 0:
                        logger.info(f"Order placed successfully: {symbol} {side} {qty}")
                        return {
                            "success": True,
                            "order_id": data.get("result", {}).get("orderId"),
                            "data": data
                        }
                    else:
                        logger.error(f"Order failed: {data.get('retMsg', 'Unknown error')}")
                        return {
                            "success": False,
                            "error": data.get("retMsg", "Unknown error"),
                            "data": data
                        }
                        
        except Exception as e:
            logger.error(f"Exception placing order: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _get_authenticated_headers(self, method: str, endpoint: str, params: Dict = None) -> Dict[str, str]:
        """Generate authenticated headers for Bybit API."""
        import hmac
        import hashlib
        import time
        
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        if params:
            query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        else:
            query_string = ""
        
        param_str = timestamp + self.api_key + recv_window + query_string
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json"
        }