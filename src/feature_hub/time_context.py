"""
Time context feature engine for temporal pattern recognition.

Features computed:
- Market session indicators
- Funding window proximity
- Economic event flags
- Temporal cyclical patterns
- Market microstructure timing
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


class TimeContextEngine:
    """
    Time context feature engine for temporal pattern analysis.
    
    Optimized for:
    - Market session detection
    - Event timing analysis
    - Cyclical pattern recognition
    - Cost-effective time calculations
    """
    
    def __init__(self):
        """Initialize time context engine."""
        
        # Market session definitions (UTC)
        self.market_sessions = {
            "asia_open": {"start": 0, "end": 9},      # 00:00-09:00 UTC
            "london_open": {"start": 7, "end": 16},   # 07:00-16:00 UTC  
            "ny_open": {"start": 13, "end": 22},      # 13:00-22:00 UTC
            "overlap_london_ny": {"start": 13, "end": 16},  # 13:00-16:00 UTC
            "low_liquidity": {"start": 22, "end": 1}, # 22:00-01:00 UTC
        }
        
        # Bybit funding times (every 8 hours at 00:00, 08:00, 16:00 UTC)
        self.funding_hours = [0, 8, 16]
        self.funding_window_minutes = 30  # 30 minutes around funding
        
        # Economic event schedule (simplified)
        # In production, this would connect to economic calendar API
        self.high_impact_hours = {
            # US market hours when major data is released
            1: [13, 14, 15],  # Monday
            2: [13, 14, 15],  # Tuesday  
            3: [13, 14, 15],  # Wednesday
            4: [13, 14, 15],  # Thursday
            5: [13, 14, 15],  # Friday
        }
        
        # Market microstructure patterns
        self.volatility_patterns = {
            "opening_vol": {"start": 0, "end": 2},    # First 2 hours of day
            "closing_vol": {"start": 21, "end": 23}, # Last 2 hours of day
            "lunch_lull": {"start": 11, "end": 13},  # Low activity period
        }
        
        # Feature cache for performance
        self.cached_features: Dict[str, Any] = {}
        self.cache_timestamp = 0
        self.cache_duration = 60  # Cache for 1 minute
    
    def get_time_features(self, timestamp: Optional[float] = None) -> Dict[str, float]:
        """
        Get comprehensive time context features.
        
        Args:
            timestamp: Unix timestamp (defaults to current time)
            
        Returns:
            Dictionary of time-based features
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Check cache first
        if timestamp - self.cache_timestamp < self.cache_duration and self.cached_features:
            return self.cached_features.copy()
        
        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            
            features = {}
            
            # 1. Basic time components
            features.update(self._get_basic_time_features(dt))
            
            # 2. Market session features
            features.update(self._get_session_features(dt))
            
            # 3. Funding window features
            features.update(self._get_funding_features(dt))
            
            # 4. Economic event proximity
            features.update(self._get_event_features(dt))
            
            # 5. Cyclical patterns
            features.update(self._get_cyclical_features(dt))
            
            # 6. Volatility timing patterns
            features.update(self._get_volatility_timing_features(dt))
            
            # Cache results
            self.cached_features = features
            self.cache_timestamp = timestamp
            
            return features
            
        except Exception as e:
            logger.warning("Error computing time context features", exception=e)
            return {}
    
    def _get_basic_time_features(self, dt: datetime) -> Dict[str, float]:
        """Get basic time component features."""
        features = {}
        
        # Raw time components
        features["hour_of_day"] = float(dt.hour)
        features["minute_of_hour"] = float(dt.minute)
        features["second_of_minute"] = float(dt.second)
        features["day_of_week"] = float(dt.weekday())  # 0=Monday, 6=Sunday
        features["day_of_month"] = float(dt.day)
        features["month_of_year"] = float(dt.month)
        
        # Cyclical encodings for smooth transitions
        # Hour of day (24-hour cycle)
        hour_radians = 2 * np.pi * dt.hour / 24
        features["hour_sin"] = float(np.sin(hour_radians))
        features["hour_cos"] = float(np.cos(hour_radians))
        
        # Minute of hour (60-minute cycle)  
        minute_radians = 2 * np.pi * dt.minute / 60
        features["minute_sin"] = float(np.sin(minute_radians))
        features["minute_cos"] = float(np.cos(minute_radians))
        
        # Day of week (7-day cycle)
        dow_radians = 2 * np.pi * dt.weekday() / 7
        features["dow_sin"] = float(np.sin(dow_radians))
        features["dow_cos"] = float(np.cos(dow_radians))
        
        # Weekend indicator
        features["is_weekend"] = float(dt.weekday() >= 5)  # Saturday=5, Sunday=6
        
        # Time since start of day (in hours)
        features["time_since_day_start"] = float(dt.hour + dt.minute / 60)
        
        return features
    
    def _get_session_features(self, dt: datetime) -> Dict[str, float]:
        """Get market session features."""
        features = {}
        
        current_hour = dt.hour
        
        # Individual session indicators
        for session_name, session_info in self.market_sessions.items():
            start_hour = session_info["start"]
            end_hour = session_info["end"]
            
            # Handle sessions that cross midnight
            if start_hour > end_hour:
                in_session = current_hour >= start_hour or current_hour < end_hour
            else:
                in_session = start_hour <= current_hour < end_hour
            
            features[f"session_{session_name}"] = float(in_session)
            
            # Distance to session start/end
            if in_session:
                # Time until session end
                if start_hour > end_hour:  # Crosses midnight
                    if current_hour >= start_hour:
                        hours_to_end = (24 - current_hour) + end_hour
                    else:
                        hours_to_end = end_hour - current_hour
                else:
                    hours_to_end = end_hour - current_hour
                
                features[f"session_{session_name}_time_to_end"] = float(hours_to_end)
            else:
                # Time until session start
                if start_hour > end_hour:  # Crosses midnight
                    if current_hour < end_hour:
                        hours_to_start = start_hour - current_hour
                    else:
                        hours_to_start = (24 - current_hour) + start_hour
                else:
                    if current_hour < start_hour:
                        hours_to_start = start_hour - current_hour
                    else:
                        hours_to_start = (24 - current_hour) + start_hour
                
                features[f"session_{session_name}_time_to_start"] = float(hours_to_start)
        
        # Session overlap indicators
        features["session_overlap_count"] = sum(
            features.get(f"session_{name}", 0) 
            for name in ["asia_open", "london_open", "ny_open"]
        )
        
        # Peak liquidity indicator (multiple sessions active)
        features["peak_liquidity"] = float(features["session_overlap_count"] >= 2)
        
        return features
    
    def _get_funding_features(self, dt: datetime) -> Dict[str, float]:
        """Get funding window features."""
        features = {}
        
        current_hour = dt.hour
        current_minute = dt.minute
        
        # Find next funding time
        next_funding_hour = None
        for funding_hour in self.funding_hours:
            if funding_hour > current_hour or (funding_hour == current_hour and current_minute < 0):
                next_funding_hour = funding_hour
                break
        
        if next_funding_hour is None:
            next_funding_hour = self.funding_hours[0] + 24  # Next day
        
        # Calculate minutes to next funding
        if next_funding_hour >= 24:
            hours_to_funding = (next_funding_hour - 24) + (24 - current_hour)
        else:
            hours_to_funding = next_funding_hour - current_hour
        
        minutes_to_funding = hours_to_funding * 60 - current_minute
        features["minutes_to_funding"] = float(minutes_to_funding)
        
        # Funding window indicator
        in_funding_window = abs(minutes_to_funding) <= self.funding_window_minutes
        features["funding_window"] = float(in_funding_window)
        
        # Funding window proximity (0 to 1, 1 = at funding time)
        if abs(minutes_to_funding) <= self.funding_window_minutes:
            proximity = 1 - abs(minutes_to_funding) / self.funding_window_minutes
            features["funding_proximity"] = float(proximity)
        else:
            features["funding_proximity"] = 0.0
        
        # Pre/post funding indicators
        features["pre_funding"] = float(0 < minutes_to_funding <= self.funding_window_minutes)
        features["post_funding"] = float(-self.funding_window_minutes <= minutes_to_funding < 0)
        
        # Funding window phase (cyclical)
        # Maps time within funding cycle (8 hours) to 0-1
        hours_since_last_funding = current_hour % 8 + current_minute / 60
        funding_phase = hours_since_last_funding / 8
        features["funding_phase"] = float(funding_phase)
        
        # Cyclical encoding of funding phase
        phase_radians = 2 * np.pi * funding_phase
        features["funding_phase_sin"] = float(np.sin(phase_radians))
        features["funding_phase_cos"] = float(np.cos(phase_radians))
        
        return features
    
    def _get_event_features(self, dt: datetime) -> Dict[str, float]:
        """Get economic event timing features."""
        features = {}
        
        day_of_week = dt.weekday() + 1  # Convert to 1=Monday format
        current_hour = dt.hour
        
        # High impact event hours
        high_impact_hours = self.high_impact_hours.get(day_of_week, [])
        features["high_impact_hour"] = float(current_hour in high_impact_hours)
        
        # Time to next high impact period
        next_impact_hour = None
        for hour in high_impact_hours:
            if hour > current_hour:
                next_impact_hour = hour
                break
        
        if next_impact_hour is None:
            # Look for next day's first high impact hour
            tomorrow_dow = (day_of_week % 7) + 1
            tomorrow_hours = self.high_impact_hours.get(tomorrow_dow, [])
            if tomorrow_hours:
                next_impact_hour = min(tomorrow_hours) + 24
        
        if next_impact_hour is not None:
            hours_to_impact = next_impact_hour - current_hour
            features["hours_to_high_impact"] = float(hours_to_impact)
        else:
            features["hours_to_high_impact"] = float(24)  # Default to 24 hours
        
        # Specific day patterns
        features["is_monday"] = float(day_of_week == 1)
        features["is_friday"] = float(day_of_week == 5)
        features["is_midweek"] = float(day_of_week in [2, 3, 4])
        
        # US market pre/post session
        features["us_premarket"] = float(10 <= current_hour < 13)  # 6-9 AM EST
        features["us_afterhours"] = float(22 <= current_hour or current_hour < 4)  # 6 PM - 12 AM EST
        
        return features
    
    def _get_cyclical_features(self, dt: datetime) -> Dict[str, float]:
        """Get cyclical pattern features."""
        features = {}
        
        # Month cyclical patterns
        month_radians = 2 * np.pi * (dt.month - 1) / 12
        features["month_sin"] = float(np.sin(month_radians))
        features["month_cos"] = float(np.cos(month_radians))
        
        # Quarter patterns
        quarter = (dt.month - 1) // 3 + 1
        features["quarter"] = float(quarter)
        quarter_radians = 2 * np.pi * (quarter - 1) / 4
        features["quarter_sin"] = float(np.sin(quarter_radians))
        features["quarter_cos"] = float(np.cos(quarter_radians))
        
        # End of month/quarter effects
        import calendar
        last_day_of_month = calendar.monthrange(dt.year, dt.month)[1]
        features["days_to_month_end"] = float(last_day_of_month - dt.day)
        features["is_month_end"] = float(dt.day >= last_day_of_month - 2)  # Last 3 days
        
        # Quarter end
        quarter_end_months = [3, 6, 9, 12]
        is_quarter_end_month = dt.month in quarter_end_months
        features["is_quarter_end_month"] = float(is_quarter_end_month)
        
        if is_quarter_end_month:
            features["is_quarter_end"] = float(dt.day >= last_day_of_month - 2)
        else:
            features["is_quarter_end"] = 0.0
        
        # Year-end effects
        features["is_december"] = float(dt.month == 12)
        features["is_january"] = float(dt.month == 1)
        
        # Holiday proximity (simplified)
        # This would be enhanced with actual holiday calendar
        features["potential_holiday_week"] = float(
            (dt.month == 12 and dt.day >= 20) or  # Christmas week
            (dt.month == 1 and dt.day <= 7) or   # New Year week
            (dt.month == 7 and dt.day <= 7)      # Independence Day week (US)
        )
        
        return features
    
    def _get_volatility_timing_features(self, dt: datetime) -> Dict[str, float]:
        """Get volatility timing pattern features."""
        features = {}
        
        current_hour = dt.hour
        
        # Volatility period indicators
        for pattern_name, pattern_info in self.volatility_patterns.items():
            start_hour = pattern_info["start"]
            end_hour = pattern_info["end"]
            
            if start_hour > end_hour:  # Crosses midnight
                in_pattern = current_hour >= start_hour or current_hour < end_hour
            else:
                in_pattern = start_hour <= current_hour < end_hour
            
            features[f"vol_pattern_{pattern_name}"] = float(in_pattern)
        
        # Market microstructure timing
        features["first_hour"] = float(current_hour == 0)
        features["last_hour"] = float(current_hour == 23)
        features["market_open_hour"] = float(current_hour in [0, 8, 13])  # Major session opens
        features["market_close_hour"] = float(current_hour in [7, 15, 22])  # Major session closes
        
        # Rollover periods (potential for increased activity)
        features["rollover_period"] = float(current_hour in [0, 8, 16])  # Funding times
        
        # Low activity periods
        features["low_activity"] = float(
            2 <= current_hour <= 6 or  # Deep night hours
            (dt.weekday() >= 5 and current_hour not in [21, 22, 23, 0])  # Weekend (except Sunday evening)
        )
        
        # News release timing (typically on the hour or half-hour)
        features["news_release_time"] = float(dt.minute in [0, 30])
        
        # Options expiry patterns (simplified - third Friday approximation)
        features["potential_expiry_day"] = float(
            dt.weekday() == 4 and  # Friday
            15 <= dt.day <= 21     # Third week of month
        )
        
        return features
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of time context features."""
        current_time = time.time()
        dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
        
        features = self.get_time_features(current_time)
        
        return {
            "current_utc_time": dt.isoformat(),
            "feature_count": len(features),
            "active_sessions": [
                name for name in self.market_sessions.keys() 
                if features.get(f"session_{name}", 0) > 0
            ],
            "funding_status": {
                "in_window": bool(features.get("funding_window", 0)),
                "minutes_to_next": features.get("minutes_to_funding", 0),
                "proximity": features.get("funding_proximity", 0)
            },
            "market_conditions": {
                "is_weekend": bool(features.get("is_weekend", 0)),
                "is_high_impact_hour": bool(features.get("high_impact_hour", 0)),
                "peak_liquidity": bool(features.get("peak_liquidity", 0)),
                "low_activity": bool(features.get("low_activity", 0))
            },
            "cached": self.cache_timestamp > 0,
            "cache_age": current_time - self.cache_timestamp
        }