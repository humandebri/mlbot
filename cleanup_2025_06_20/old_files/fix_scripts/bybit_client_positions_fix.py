"""
Fix for BybitRESTClient get_open_positions method.
Add this code to the existing BybitRESTClient class.
"""

async def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get open positions with proper error handling."""
    try:
        # CRITICAL: Ensure session exists
        if not self.session:
            self.session = aiohttp.ClientSession()
            logger.info("Created new aiohttp session for positions")
        
        async with self.request_semaphore:
            await self._wait_for_rate_limit()
            
            url = urljoin(self.base_url, "/v5/position/list")
            params = {
                "category": "linear",
                "settleCoin": "USDT"
            }
            if symbol:
                params["symbol"] = symbol
            
            # Get auth headers
            headers = self._get_auth_headers("GET", "/v5/position/list", params)
            
            # CRITICAL: Add None check for headers
            if not headers:
                logger.error("Failed to generate auth headers for positions")
                return []
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # CRITICAL: Check if data is None
                    if data is None:
                        logger.error("Received None response from positions API")
                        return []
                    
                    if data.get("retCode") == 0:
                        result = data.get("result", {})
                        # CRITICAL: Safe navigation for nested dict
                        if isinstance(result, dict):
                            return result.get("list", [])
                        else:
                            logger.error(f"Unexpected result format: {type(result)}")
                            return []
                    else:
                        logger.error(f"API error getting positions: {data.get('retMsg')}")
                else:
                    logger.error(f"HTTP error {response.status} getting positions")
                    # Log response body for debugging
                    try:
                        error_body = await response.text()
                        logger.error(f"Error response: {error_body[:200]}")
                    except:
                        pass
                        
    except aiohttp.ClientError as e:
        logger.error(f"Network error fetching positions: {e}")
        increment_counter(ERRORS_TOTAL, component="rest_api", error_type="network_error")
    except Exception as e:
        logger.error(f"Unexpected error fetching positions: {e}", exc_info=True)
        increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
    
    return []

def _get_auth_headers(self, method: str, path: str, params: Optional[Dict] = None) -> Optional[Dict[str, str]]:
    """Generate authentication headers with error handling."""
    try:
        # Check if API credentials are available
        if not self.api_key or not self.api_secret:
            logger.error("API credentials not configured")
            return None
        
        timestamp = str(int(time.time() * 1000))
        recv_window = str(5000)
        
        # Prepare parameters
        param_str = ""
        if params:
            sorted_params = sorted(params.items())
            param_str = "&".join([f"{k}={v}" for k, v in sorted_params])
        
        # Create signature payload
        raw_data = f"{timestamp}{self.api_key}{recv_window}{param_str}"
        
        # Generate signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            raw_data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }
        
        return headers
        
    except Exception as e:
        logger.error(f"Error generating auth headers: {e}", exc_info=True)
        return None