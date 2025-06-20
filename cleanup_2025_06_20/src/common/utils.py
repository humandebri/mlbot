"""
Common utility functions for the trading bot.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .exceptions import ValidationError
from .logging import get_logger

logger = get_logger(__name__)


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    if value is None or value == '':
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert {value} to float, using default {default}")
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Int value or default
    """
    if value is None or value == '':
        return default
    
    try:
        return int(float(value))  # Handle string numbers like "5.0"
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert {value} to int, using default {default}")
        return default


def safe_decimal(value: Any, precision: int = 8, default: Optional[Decimal] = None) -> Decimal:
    """
    Safely convert value to Decimal with specified precision.
    
    Args:
        value: Value to convert
        precision: Number of decimal places
        default: Default value if conversion fails
        
    Returns:
        Decimal value or default
    """
    if default is None:
        default = Decimal('0')
    
    if value is None or value == '':
        return default
    
    try:
        # Convert to Decimal and round to precision
        decimal_value = Decimal(str(value))
        return decimal_value.quantize(Decimal('0.' + '0' * precision), rounding=ROUND_DOWN)
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert {value} to Decimal, using default {default}")
        return default


def format_currency(amount: Union[float, Decimal], currency: str = "USD", precision: int = 2) -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        currency: Currency symbol
        precision: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    if isinstance(amount, Decimal):
        amount = float(amount)
    
    if currency == "USD":
        return f"${amount:,.{precision}f}"
    else:
        return f"{amount:,.{precision}f} {currency}"


def format_percentage(value: Union[float, Decimal], precision: int = 2) -> str:
    """
    Format value as percentage string.
    
    Args:
        value: Value to format (0.05 = 5%)
        precision: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if isinstance(value, Decimal):
        value = float(value)
    
    return f"{value * 100:.{precision}f}%"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def get_utc_timestamp() -> int:
    """Get current UTC timestamp in milliseconds."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def get_utc_datetime() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """
    Format datetime as string.
    
    Args:
        dt: Datetime to format
        format_str: Format string
        
    Returns:
        Formatted datetime string
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.strftime(format_str)


def parse_datetime(date_string: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    Parse datetime string.
    
    Args:
        date_string: Datetime string to parse
        format_str: Format string
        
    Returns:
        Parsed datetime
    """
    try:
        dt = datetime.strptime(date_string, format_str)
        return dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValidationError(f"Failed to parse datetime '{date_string}': {e}")


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')
    
    return (new_value - old_value) / old_value


def round_to_precision(value: float, precision: int) -> float:
    """
    Round value to specified precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        
    Returns:
        Rounded value
    """
    multiplier = 10 ** precision
    return round(value * multiplier) / multiplier


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max.
    
    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def normalize_symbol(symbol: str) -> str:
    """
    Normalize trading symbol format.
    
    Args:
        symbol: Symbol to normalize
        
    Returns:
        Normalized symbol
    """
    return symbol.upper().replace("-", "").replace("_", "").replace("/", "")


def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate unique ID.
    
    Args:
        prefix: Prefix for the ID
        length: Length of the random part
        
    Returns:
        Generated ID
    """
    timestamp = str(int(time.time() * 1000))
    random_part = hashlib.md5(timestamp.encode()).hexdigest()[:length]
    
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def create_hash(data: Any) -> str:
    """
    Create MD5 hash of data.
    
    Args:
        data: Data to hash
        
    Returns:
        MD5 hash string
    """
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    elif not isinstance(data, str):
        data = str(data)
    
    return hashlib.md5(data.encode()).hexdigest()


def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(data: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten nested dictionary.
    
    Args:
        data: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def chunk_list(items: List, chunk_size: int) -> List[List]:
    """
    Split list into chunks.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i:i + chunk_size])
    return chunks


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_json_file(file_path: Union[str, Path]) -> Dict:
    """
    Load JSON file safely.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValidationError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in file {file_path}: {e}")


def save_json_file(data: Dict, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
    """
    try:
        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        raise ValidationError(f"Failed to save JSON file {file_path}: {e}")


async def run_with_timeout(coro, timeout_seconds: float):
    """
    Run coroutine with timeout.
    
    Args:
        coro: Coroutine to run
        timeout_seconds: Timeout in seconds
        
    Returns:
        Result of coroutine
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")


def memory_efficient_batch_processor(
    items: List[Any],
    batch_size: int = 100,
    processor_func: callable = None
) -> List[Any]:
    """
    Process large lists in memory-efficient batches.
    
    Args:
        items: Items to process
        batch_size: Size of each batch
        processor_func: Function to process each batch
        
    Returns:
        Processed results
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        if processor_func:
            batch_result = processor_func(batch)
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        else:
            results.extend(batch)
    
    return results


class CircularBuffer:
    """Memory-efficient circular buffer implementation."""
    
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.buffer = [None] * maxsize
        self.head = 0
        self.tail = 0
        self.size = 0
    
    def append(self, item: Any) -> None:
        """Add item to buffer."""
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.maxsize
        
        if self.size < self.maxsize:
            self.size += 1
        else:
            self.head = (self.head + 1) % self.maxsize
    
    def get_items(self) -> List[Any]:
        """Get all items in buffer."""
        if self.size == 0:
            return []
        
        if self.head < self.tail:
            return self.buffer[self.head:self.tail]
        else:
            return self.buffer[self.head:] + self.buffer[:self.tail]
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.head = 0
        self.tail = 0
        self.size = 0
    
    def __len__(self) -> int:
        return self.size
    
    def __bool__(self) -> bool:
        return self.size > 0