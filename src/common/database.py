"""Database connections and utilities for Redis and DuckDB."""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import duckdb
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)

# Global connection instances
_redis_pool: Optional[ConnectionPool] = None
_redis_client: Optional[Redis] = None
_duckdb_conn: Optional[duckdb.DuckDBPyConnection] = None


async def init_redis() -> Redis:
    """Initialize Redis connection pool and client."""
    global _redis_pool, _redis_client
    
    if _redis_client is not None:
        return _redis_client
    
    try:
        _redis_pool = ConnectionPool(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            password=settings.redis.password,
            max_connections=settings.redis.max_connections,
            retry_on_timeout=settings.redis.retry_on_timeout,
            socket_timeout=settings.redis.socket_timeout,
            socket_connect_timeout=settings.redis.socket_connect_timeout,
            decode_responses=True,
        )
        
        _redis_client = Redis(connection_pool=_redis_pool)
        
        # Test connection
        await _redis_client.ping()
        logger.info("Redis connection established")
        
        return _redis_client
        
    except Exception as e:
        logger.error("Failed to initialize Redis", exception=e)
        raise


async def get_redis_client() -> Redis:
    """Get Redis client instance."""
    if _redis_client is None:
        return await init_redis()
    return _redis_client


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis_pool, _redis_client
    
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis connection closed")
    
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None


def init_duckdb() -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB connection."""
    global _duckdb_conn
    
    if _duckdb_conn is not None:
        return _duckdb_conn
    
    try:
        _duckdb_conn = duckdb.connect(
            database=settings.duckdb.database_path,
            config={
                "memory_limit": settings.duckdb.memory_limit,
                "threads": settings.duckdb.threads,
            }
        )
        
        # Configure DuckDB extensions and settings
        _duckdb_conn.execute("INSTALL httpfs")
        _duckdb_conn.execute("LOAD httpfs")
        _duckdb_conn.execute("INSTALL parquet")
        _duckdb_conn.execute("LOAD parquet")
        
        # Test connection
        _duckdb_conn.execute("SELECT 1")
        logger.info("DuckDB connection established")
        
        return _duckdb_conn
        
    except Exception as e:
        logger.error("Failed to initialize DuckDB", exception=e)
        raise


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Get DuckDB connection instance."""
    if _duckdb_conn is None:
        return init_duckdb()
    return _duckdb_conn


def close_duckdb() -> None:
    """Close DuckDB connection."""
    global _duckdb_conn
    
    if _duckdb_conn:
        _duckdb_conn.close()
        _duckdb_conn = None
        logger.info("DuckDB connection closed")


def create_trading_tables() -> None:
    """Create trading history tables if they don't exist."""
    conn = get_duckdb_connection()
    
    # Create trades table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id VARCHAR PRIMARY KEY,
            position_id VARCHAR,
            symbol VARCHAR NOT NULL,
            side VARCHAR NOT NULL,
            order_type VARCHAR NOT NULL,
            quantity DOUBLE NOT NULL,
            price DOUBLE NOT NULL,
            executed_at TIMESTAMP NOT NULL,
            fees DOUBLE DEFAULT 0,
            status VARCHAR DEFAULT 'pending',
            metadata JSON
        )
    """)
    
    # Create positions table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            position_id VARCHAR PRIMARY KEY,
            symbol VARCHAR NOT NULL,
            side VARCHAR NOT NULL,
            entry_price DOUBLE NOT NULL,
            exit_price DOUBLE,
            quantity DOUBLE NOT NULL,
            stop_loss DOUBLE,
            take_profit DOUBLE,
            opened_at TIMESTAMP NOT NULL,
            closed_at TIMESTAMP,
            pnl DOUBLE,
            fees DOUBLE DEFAULT 0,
            status VARCHAR DEFAULT 'open',
            metadata JSON
        )
    """)
    
    # Create performance table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS performance (
            date DATE PRIMARY KEY,
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            total_pnl DOUBLE,
            total_fees DOUBLE,
            max_drawdown DOUBLE,
            sharpe_ratio DOUBLE,
            win_rate DOUBLE,
            avg_win DOUBLE,
            avg_loss DOUBLE
        )
    """)
    
    conn.commit()
    logger.info("Trading tables created successfully")


def save_trade(
    trade_id: str,
    position_id: str,
    symbol: str,
    side: str,
    order_type: str,
    quantity: float,
    price: float,
    fees: float = 0,
    status: str = "executed",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save trade to database."""
    conn = get_duckdb_connection()
    
    try:
        conn.execute("""
            INSERT INTO trades (
                trade_id, position_id, symbol, side, order_type,
                quantity, price, executed_at, fees, status, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NOW(), ?, ?, ?)
        """, (
            trade_id, position_id, symbol, side, order_type,
            quantity, price, fees, status, json.dumps(metadata or {})
        ))
        conn.commit()
        logger.info(f"Trade saved: {trade_id}")
    except Exception as e:
        logger.error(f"Failed to save trade: {e}")
        conn.rollback()


def save_position(
    position_id: str,
    symbol: str,
    side: str,
    entry_price: float,
    quantity: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save position to database."""
    conn = get_duckdb_connection()
    
    try:
        conn.execute("""
            INSERT INTO positions (
                position_id, symbol, side, entry_price, quantity,
                stop_loss, take_profit, opened_at, status, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NOW(), 'open', ?)
        """, (
            position_id, symbol, side, entry_price, quantity,
            stop_loss, take_profit, json.dumps(metadata or {})
        ))
        conn.commit()
        logger.info(f"Position saved: {position_id}")
    except Exception as e:
        logger.error(f"Failed to save position: {e}")
        conn.rollback()


def update_position_close(
    position_id: str,
    exit_price: float,
    pnl: float,
    fees: float = 0
) -> None:
    """Update position when closed."""
    conn = get_duckdb_connection()
    
    try:
        conn.execute("""
            UPDATE positions 
            SET exit_price = ?, pnl = ?, fees = ?, 
                closed_at = NOW(), status = 'closed'
            WHERE position_id = ?
        """, (exit_price, pnl, fees, position_id))
        conn.commit()
        logger.info(f"Position closed: {position_id}, PnL: {pnl}")
    except Exception as e:
        logger.error(f"Failed to update position: {e}")
        conn.rollback()


class RedisStreams:
    """Redis Streams helper for real-time data."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    async def add_message(
        self,
        stream: str,
        data: Dict[str, Any],
        maxlen: Optional[int] = None,
    ) -> str:
        """Add message to Redis stream."""
        try:
            message_id = await self.redis.xadd(
                stream,
                data,
                maxlen=maxlen or settings.redis.stream_maxlen,
                approximate=True,
            )
            return message_id
        except Exception as e:
            logger.error(f"Failed to add message to stream {stream}", exception=e)
            raise
    
    async def read_messages(
        self,
        streams: Dict[str, str],
        count: Optional[int] = None,
        block: Optional[int] = None,
    ) -> List[tuple]:
        """Read messages from Redis streams."""
        try:
            messages = await self.redis.xread(
                streams,
                count=count or settings.redis.batch_size,
                block=block,
            )
            return messages
        except Exception as e:
            logger.error("Failed to read messages from streams", exception=e)
            raise
    
    async def create_consumer_group(
        self,
        stream: str,
        group: str,
        consumer_id: str = "0",
    ) -> bool:
        """Create consumer group for stream."""
        try:
            await self.redis.xgroup_create(
                stream, group, id=consumer_id, mkstream=True
            )
            logger.info(f"Created consumer group {group} for stream {stream}")
            return True
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group {group} already exists for stream {stream}")
                return True
            logger.error(f"Failed to create consumer group {group}", exception=e)
            raise
        except Exception as e:
            logger.error(f"Failed to create consumer group {group}", exception=e)
            raise
    
    async def read_group_messages(
        self,
        group: str,
        consumer: str,
        streams: Dict[str, str],
        count: Optional[int] = None,
        block: Optional[int] = None,
    ) -> List[tuple]:
        """Read messages as part of consumer group."""
        try:
            messages = await self.redis.xreadgroup(
                group,
                consumer,
                streams,
                count=count or settings.redis.batch_size,
                block=block,
            )
            return messages
        except Exception as e:
            logger.error("Failed to read group messages", exception=e)
            raise
    
    async def ack_message(self, stream: str, group: str, message_id: str) -> None:
        """Acknowledge message processing."""
        try:
            await self.redis.xack(stream, group, message_id)
        except Exception as e:
            logger.error(f"Failed to ack message {message_id}", exception=e)
            raise


class DuckDBHelper:
    """DuckDB helper for time-series data operations."""
    
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
    
    def create_tables(self) -> None:
        """Create necessary tables."""
        # Raw market data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TIMESTAMPTZ,
                symbol VARCHAR,
                data_type VARCHAR,
                data JSON,
                PRIMARY KEY (timestamp, symbol, data_type)
            )
        """)
        
        # Processed features table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                timestamp TIMESTAMPTZ,
                symbol VARCHAR,
                features JSON,
                PRIMARY KEY (timestamp, symbol)
            )
        """)
        
        # Trading signals table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                timestamp TIMESTAMPTZ,
                symbol VARCHAR,
                signal_type VARCHAR,
                strength DOUBLE,
                delta DOUBLE,
                lookahead INTEGER,
                features JSON,
                PRIMARY KEY (timestamp, symbol, signal_type)
            )
        """)
        
        # Orders table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMPTZ,
                symbol VARCHAR,
                side VARCHAR,
                order_type VARCHAR,
                qty DOUBLE,
                price DOUBLE,
                status VARCHAR,
                filled_qty DOUBLE,
                filled_price DOUBLE,
                fee DOUBLE,
                created_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ
            )
        """)
        
        # Positions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                timestamp TIMESTAMPTZ,
                symbol VARCHAR,
                size DOUBLE,
                entry_price DOUBLE,
                mark_price DOUBLE,
                pnl DOUBLE,
                PRIMARY KEY (timestamp, symbol)
            )
        """)
        
        logger.info("DuckDB tables created/verified")
    
    def insert_market_data(
        self,
        timestamp: str,
        symbol: str,
        data_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Insert market data."""
        self.conn.execute(
            "INSERT INTO market_data VALUES (?, ?, ?, ?)",
            [timestamp, symbol, data_type, json.dumps(data)]
        )
    
    def insert_features(
        self,
        timestamp: str,
        symbol: str,
        features: Dict[str, float],
    ) -> None:
        """Insert processed features."""
        self.conn.execute(
            "INSERT INTO features VALUES (?, ?, ?)",
            [timestamp, symbol, json.dumps(features)]
        )
    
    def get_features(
        self,
        symbol: str,
        start_time: str,
        end_time: str,
    ) -> List[Dict[str, Any]]:
        """Get features for a time range."""
        result = self.conn.execute(
            """
            SELECT timestamp, features
            FROM features
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
            """,
            [symbol, start_time, end_time]
        ).fetchall()
        
        return [
            {"timestamp": row[0], "features": json.loads(row[1])}
            for row in result
        ]
    
    def export_to_parquet(
        self,
        table: str,
        filename: str,
        where_clause: str = "",
    ) -> None:
        """Export table to Parquet file."""
        query = f"COPY (SELECT * FROM {table}"
        if where_clause:
            query += f" WHERE {where_clause}"
        query += f") TO '{filename}' (FORMAT PARQUET, COMPRESSION '{settings.duckdb.parquet_compression}')"
        
        self.conn.execute(query)
        logger.info(f"Exported {table} to {filename}")


@asynccontextmanager
async def get_database_connections() -> AsyncGenerator[tuple[Redis, duckdb.DuckDBPyConnection], None]:
    """Context manager for database connections."""
    redis_client = None
    duckdb_conn = None
    
    try:
        # Initialize connections
        redis_client = await init_redis()
        duckdb_conn = init_duckdb()
        
        # Create tables
        duckdb_helper = DuckDBHelper(duckdb_conn)
        duckdb_helper.create_tables()
        
        yield redis_client, duckdb_conn
        
    finally:
        # Cleanup connections
        if redis_client:
            await close_redis()
        if duckdb_conn:
            close_duckdb()


async def init_databases() -> tuple[Redis, duckdb.DuckDBPyConnection]:
    """Initialize both databases."""
    redis_client = await init_redis()
    duckdb_conn = init_duckdb()
    
    # Create DuckDB tables
    duckdb_helper = DuckDBHelper(duckdb_conn)
    duckdb_helper.create_tables()
    
    return redis_client, duckdb_conn


async def close_databases() -> None:
    """Close all database connections."""
    await close_redis()
    close_duckdb()


class RedisManager:
    """Redis connection manager for trading system."""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        self.redis = await get_redis_client()
        logger.info("RedisManager connected to Redis")
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if not self.redis:
            return None
        try:
            return await self.redis.get(key)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def xread(self, streams: Dict[str, str], count: int = None, block: int = None) -> List:
        """Read from Redis streams."""
        if not self.redis:
            return []
        try:
            return await self.redis.xread(streams, count=count, block=block)
        except Exception as e:
            logger.error(f"Redis xread error: {e}")
            return []
    
    async def xrevrange(self, stream: str, start: str = "+", end: str = "-", count: int = None) -> List:
        """Read from Redis stream in reverse order."""
        if not self.redis:
            return []
        try:
            return await self.redis.xrevrange(stream, start, end, count=count)
        except Exception as e:
            logger.error(f"Redis xrevrange error: {e}")
            return []
    
    async def ping(self) -> bool:
        """Ping Redis to check connection."""
        if not self.redis:
            return False
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping error: {e}")
            return False
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("RedisManager closed Redis connection")