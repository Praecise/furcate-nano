# ============================================================================
# furcate_nano/storage.py
"""
Complete storage management system with batch processing, compression, 
and backup mechanisms for environmental data.
"""

import asyncio
import logging
import sqlite3
import json
import time
import gzip
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue

logger = logging.getLogger(__name__)

class StorageType(Enum):
    """Storage backend types."""
    SQLITE = "sqlite"
    TIMESERIES = "timeseries"
    FILE_BASED = "file_based"

class CompressionType(Enum):
    """Compression methods."""
    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"

@dataclass
class StorageMetrics:
    """Storage system metrics."""
    total_records: int = 0
    total_size_bytes: int = 0
    records_per_hour: float = 0.0
    average_record_size: float = 0.0
    compression_ratio: float = 1.0
    database_files: int = 0
    oldest_record: Optional[datetime] = None
    newest_record: Optional[datetime] = None
    write_operations: int = 0
    read_operations: int = 0
    failed_operations: int = 0
    backup_count: int = 0
    last_backup: Optional[datetime] = None

class DatabaseManager:
    """SQLite database manager with connection pooling."""
    
    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self._connection_pool = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._initialized = False
        
    def initialize(self):
        """Initialize database and connection pool."""
        if self._initialized:
            return
        
        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create database if it doesn't exist
        conn = sqlite3.connect(self.db_path)
        self._create_tables(conn)
        conn.close()
        
        # Fill connection pool
        for _ in range(self.max_connections):
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._connection_pool.put(conn)
        
        self._initialized = True
        logger.info(f"📦 Database initialized: {self.db_path}")
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables."""
        # Environmental data table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS environmental_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                sensor_data TEXT NOT NULL,
                ml_analysis TEXT,
                quality_score REAL,
                location TEXT,
                metadata TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        
        # Alerts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metadata TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        
        # System events table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT NOT NULL,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        
        # ML model data table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_model_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                training_data TEXT,
                performance_metrics TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        
        # Create indexes for better performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_env_timestamp ON environmental_data(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_env_device ON environmental_data(device_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp)")
        
        conn.commit()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get connection from pool."""
        if not self._initialized:
            self.initialize()
        
        try:
            return self._connection_pool.get(timeout=5.0)
        except queue.Empty:
            # Create emergency connection if pool is exhausted
            logger.warning("Connection pool exhausted, creating emergency connection")
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            return conn
    
    def return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool."""
        try:
            if self._connection_pool.qsize() < self.max_connections:
                self._connection_pool.put(conn, timeout=1.0)
            else:
                conn.close()
        except queue.Full:
            conn.close()
    
    def close_all(self):
        """Close all connections."""
        while not self._connection_pool.empty():
            try:
                conn = self._connection_pool.get_nowait()
                conn.close()
            except queue.Empty:
                break

class CompressionManager:
    """Handles data compression and decompression."""
    
    def __init__(self, compression_type: CompressionType = CompressionType.GZIP):
        self.compression_type = compression_type
    
    def compress_data(self, data: str) -> bytes:
        """Compress string data."""
        if self.compression_type == CompressionType.NONE:
            return data.encode('utf-8')
        
        elif self.compression_type == CompressionType.GZIP:
            return gzip.compress(data.encode('utf-8'))
        
        elif self.compression_type == CompressionType.LZMA:
            import lzma
            return lzma.compress(data.encode('utf-8'))
        
        else:
            return data.encode('utf-8')
    
    def decompress_data(self, data: bytes) -> str:
        """Decompress data back to string."""
        if self.compression_type == CompressionType.NONE:
            return data.decode('utf-8')
        
        elif self.compression_type == CompressionType.GZIP:
            return gzip.decompress(data).decode('utf-8')
        
        elif self.compression_type == CompressionType.LZMA:
            import lzma
            return lzma.decompress(data).decode('utf-8')
        
        else:
            return data.decode('utf-8')

class BackupManager:
    """Manages database backups and rotation."""
    
    def __init__(self, storage_path: str, backup_interval: int = 3600, max_backups: int = 24):
        self.storage_path = Path(storage_path)
        self.backup_interval = backup_interval  # seconds
        self.max_backups = max_backups
        self.backup_path = self.storage_path / "backups"
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.last_backup = None
        
    def should_backup(self) -> bool:
        """Check if backup is needed."""
        if not self.last_backup:
            return True
        
        return (time.time() - self.last_backup) >= self.backup_interval
    
    async def create_backup(self, db_manager: DatabaseManager) -> bool:
        """Create database backup."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_path / f"furcate_nano_{timestamp}.db"
            
            # Copy database file
            shutil.copy2(db_manager.db_path, backup_file)
            
            # Compress backup
            compressed_backup = backup_file.with_suffix('.db.gz')
            with open(backup_file, 'rb') as f_in:
                with gzip.open(compressed_backup, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove uncompressed backup
            backup_file.unlink()
            
            self.last_backup = time.time()
            logger.info(f"📦 Database backup created: {compressed_backup}")
            
            # Clean old backups
            await self._cleanup_old_backups()
            
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False
    
    async def _cleanup_old_backups(self):
        """Remove old backup files."""
        try:
            backup_files = sorted(
                self.backup_path.glob("furcate_nano_*.db.gz"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Keep only the most recent backups
            for backup_file in backup_files[self.max_backups:]:
                backup_file.unlink()
                logger.debug(f"Removed old backup: {backup_file}")
                
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def get_backup_list(self) -> List[Dict[str, Any]]:
        """Get list of available backups."""
        backups = []
        
        for backup_file in sorted(
            self.backup_path.glob("furcate_nano_*.db.gz"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        ):
            stat = backup_file.stat()
            backups.append({
                "filename": backup_file.name,
                "path": str(backup_file),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_mtime),
                "age_hours": (time.time() - stat.st_mtime) / 3600
            })
        
        return backups

class StorageManager:
    """Complete storage management system with batch processing and compression."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device_id = config.get("device_id", "unknown")
        
        # Storage configuration
        self.storage_path = Path(config.get("storage_path", "./data"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Database setup
        self.db_path = self.storage_path / "furcate_nano.db"
        self.db_manager = DatabaseManager(str(self.db_path))
        
        # Batch processing
        self.batch_size = config.get("batch_size", 100)
        self.batch_timeout = config.get("batch_timeout", 30)  # seconds
        self.write_batch = []
        self.batch_lock = asyncio.Lock()
        self.batch_task = None
        
        # Compression
        compression_type = config.get("compression", "gzip")
        self.compression_manager = CompressionManager(
            CompressionType(compression_type) if compression_type != "none" else CompressionType.NONE
        )
        
        # Backup management
        self.backup_manager = BackupManager(
            str(self.storage_path),
            backup_interval=config.get("backup_interval", 3600),
            max_backups=config.get("max_backups", 24)
        )
        
        # Data retention
        self.retention_days = config.get("retention_days", 30)
        self.cleanup_interval = config.get("cleanup_interval", 86400)  # 24 hours
        self.last_cleanup = None
        
        # Metrics
        self.metrics = StorageMetrics()
        
        # Statistics tracking
        self.stats = {
            "storage_operations": 0,
            "storage_errors": 0,
            "bytes_stored": 0,
            "compression_savings": 0
        }
        
        logger.info("📦 Storage Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize storage system."""
        try:
            # Initialize database
            self.db_manager.initialize()
            
            # Start batch processing task
            self.batch_task = asyncio.create_task(self._batch_processing_loop())
            
            # Update metrics
            await self._update_metrics()
            
            logger.info("✅ Storage system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Storage initialization failed: {e}")
            return False
    
    async def store_environmental_record(self, record: Dict[str, Any]) -> bool:
        """Store environmental data record with batch processing."""
        try:
            # Add to batch for efficient writing
            storage_record = {
                'type': 'environmental',
                'data': record,
                'timestamp': time.time(),
                'size_bytes': len(self._serialize_for_storage(record))
            }
            
            async with self.batch_lock:
                self.write_batch.append(storage_record)
                
                # If batch is full, process immediately
                if len(self.write_batch) >= self.batch_size:
                    await self._flush_batch()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store environmental record: {e}")
            self.stats["storage_errors"] += 1
            return False
    
    async def store_alert(self, alert: Dict[str, Any], device_id: str) -> bool:
        """Store alert record."""
        try:
            alert_record = {
                'type': 'alert',
                'data': {
                    'device_id': device_id,
                    'alert': alert,
                    'timestamp': datetime.now().isoformat()
                },
                'timestamp': time.time(),
                'size_bytes': len(self._serialize_for_storage(alert))
            }
            
            async with self.batch_lock:
                self.write_batch.append(alert_record)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
            self.stats["storage_errors"] += 1
            return False
    
    async def store_system_event(self, event_type: str, event_data: Dict[str, Any], device_id: str) -> bool:
        """Store system event."""
        try:
            event_record = {
                'type': 'system_event',
                'data': {
                    'device_id': device_id,
                    'event_type': event_type,
                    'event_data': event_data,
                    'timestamp': datetime.now().isoformat()
                },
                'timestamp': time.time(),
                'size_bytes': len(self._serialize_for_storage(event_data))
            }
            
            async with self.batch_lock:
                self.write_batch.append(event_record)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store system event: {e}")
            self.stats["storage_errors"] += 1
            return False
    
    async def store_ml_model_data(self, model_name: str, version: str, 
                                 training_data: Dict[str, Any], 
                                 performance_metrics: Dict[str, Any]) -> bool:
        """Store ML model training data and metrics."""
        try:
            model_record = {
                'type': 'ml_model',
                'data': {
                    'model_name': model_name,
                    'version': version,
                    'training_data': training_data,
                    'performance_metrics': performance_metrics,
                    'timestamp': datetime.now().isoformat()
                },
                'timestamp': time.time(),
                'size_bytes': len(self._serialize_for_storage(training_data)) + 
                             len(self._serialize_for_storage(performance_metrics))
            }
            
            async with self.batch_lock:
                self.write_batch.append(model_record)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store ML model data: {e}")
            self.stats["storage_errors"] += 1
            return False
    
    async def _flush_batch(self):
        """Flush write batch to database."""
        if not self.write_batch:
            return
        
        try:
            batch_to_process = self.write_batch.copy()
            self.write_batch.clear()
            
            conn = self.db_manager.get_connection()
            
            try:
                conn.execute("BEGIN TRANSACTION")
                
                for record in batch_to_process:
                    await self._write_record_to_db(conn, record)
                
                conn.execute("COMMIT")
                
                # Update metrics
                self.metrics.write_operations += 1
                self.stats["storage_operations"] += len(batch_to_process)
                
                total_size = sum(r['size_bytes'] for r in batch_to_process)
                self.stats["bytes_stored"] += total_size
                
                logger.debug(f"📦 Flushed batch of {len(batch_to_process)} records ({total_size} bytes)")
                
            except Exception as e:
                conn.execute("ROLLBACK")
                raise e
            
            finally:
                self.db_manager.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Batch flush failed: {e}")
            self.stats["storage_errors"] += 1
            
            # Put records back in batch for retry
            async with self.batch_lock:
                self.write_batch.extend(batch_to_process)
    
    async def _write_record_to_db(self, conn: sqlite3.Connection, record: Dict[str, Any]):
        """Write individual record to database."""
        record_type = record['type']
        data = record['data']
        timestamp = record['timestamp']
        
        if record_type == 'environmental':
            conn.execute("""
                INSERT INTO environmental_data 
                (device_id, timestamp, sensor_data, ml_analysis, quality_score, location, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get('device_id', self.device_id),
                timestamp,
                self._serialize_for_storage(data.get('sensor_data', {})),
                self._serialize_for_storage(data.get('ml_analysis', {})),
                data.get('quality_score', 0.0),
                self._serialize_for_storage(data.get('location', {})),
                self._serialize_for_storage(data.get('metadata', {}))
            ))
            
        elif record_type == 'alert':
            alert = data['alert']
            conn.execute("""
                INSERT INTO alerts 
                (device_id, timestamp, alert_type, severity, message, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                data['device_id'],
                timestamp,
                alert.get('type', 'unknown'),
                alert.get('severity', 'info'),
                alert.get('message', ''),
                self._serialize_for_storage(alert.get('metadata', {}))
            ))
            
        elif record_type == 'system_event':
            conn.execute("""
                INSERT INTO system_events 
                (device_id, timestamp, event_type, event_data)
                VALUES (?, ?, ?, ?)
            """, (
                data['device_id'],
                timestamp,
                data['event_type'],
                self._serialize_for_storage(data['event_data'])
            ))
            
        elif record_type == 'ml_model':
            conn.execute("""
                INSERT INTO ml_model_data 
                (model_name, version, training_data, performance_metrics)
                VALUES (?, ?, ?, ?)
            """, (
                data['model_name'],
                data['version'],
                self._serialize_for_storage(data['training_data']),
                self._serialize_for_storage(data['performance_metrics'])
            ))
    
    async def _batch_processing_loop(self):
        """Background task for batch processing."""
        while True:
            try:
                await asyncio.sleep(self.batch_timeout)
                
                async with self.batch_lock:
                    if self.write_batch:
                        await self._flush_batch()
                
                # Perform maintenance tasks
                await self._perform_maintenance()
                
            except asyncio.CancelledError:
                # Final batch flush on shutdown
                async with self.batch_lock:
                    if self.write_batch:
                        await self._flush_batch()
                break
            except Exception as e:
                logger.error(f"Batch processing loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _perform_maintenance(self):
        """Perform periodic maintenance tasks."""
        try:
            # Update metrics
            await self._update_metrics()
            
            # Check if cleanup is needed
            if self._should_cleanup():
                await self._cleanup_old_data()
            
            # Check if backup is needed
            if self.backup_manager.should_backup():
                await self.backup_manager.create_backup(self.db_manager)
                
        except Exception as e:
            logger.error(f"Maintenance task failed: {e}")
    
    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        if not self.last_cleanup:
            return True
        
        return (time.time() - self.last_cleanup) >= self.cleanup_interval
    
    async def _cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        if self.retention_days <= 0:
            return  # Retention disabled
        
        try:
            cutoff_time = time.time() - (self.retention_days * 86400)
            
            conn = self.db_manager.get_connection()
            
            try:
                # Clean up old environmental data
                cursor = conn.execute(
                    "DELETE FROM environmental_data WHERE timestamp < ?",
                    (cutoff_time,)
                )
                env_deleted = cursor.rowcount
                
                # Clean up old alerts
                cursor = conn.execute(
                    "DELETE FROM alerts WHERE timestamp < ?",
                    (cutoff_time,)
                )
                alerts_deleted = cursor.rowcount
                
                # Clean up old system events
                cursor = conn.execute(
                    "DELETE FROM system_events WHERE timestamp < ?",
                    (cutoff_time,)
                )
                events_deleted = cursor.rowcount
                
                conn.commit()
                
                # Vacuum database to reclaim space
                conn.execute("VACUUM")
                
                self.last_cleanup = time.time()
                
                total_deleted = env_deleted + alerts_deleted + events_deleted
                if total_deleted > 0:
                    logger.info(
                        f"🧹 Cleaned up {total_deleted} old records "
                        f"(env: {env_deleted}, alerts: {alerts_deleted}, events: {events_deleted})"
                    )
                
            finally:
                self.db_manager.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    async def _update_metrics(self):
        """Update storage metrics."""
        try:
            conn = self.db_manager.get_connection()
            
            try:
                # Count total records
                cursor = conn.execute("SELECT COUNT(*) FROM environmental_data")
                self.metrics.total_records = cursor.fetchone()[0]
                
                # Get oldest and newest records
                cursor = conn.execute(
                    "SELECT MIN(timestamp), MAX(timestamp) FROM environmental_data"
                )
                min_time, max_time = cursor.fetchone()
                
                if min_time:
                    self.metrics.oldest_record = datetime.fromtimestamp(min_time)
                if max_time:
                    self.metrics.newest_record = datetime.fromtimestamp(max_time)
                
                # Calculate records per hour
                if min_time and max_time and max_time > min_time:
                    time_span_hours = (max_time - min_time) / 3600
                    self.metrics.records_per_hour = self.metrics.total_records / time_span_hours
                
                # Get database file size
                db_stat = Path(self.db_manager.db_path).stat()
                self.metrics.total_size_bytes = db_stat.st_size
                
                # Calculate average record size
                if self.metrics.total_records > 0:
                    self.metrics.average_record_size = (
                        self.metrics.total_size_bytes / self.metrics.total_records
                    )
                
                # Count backup files
                backups = self.backup_manager.get_backup_list()
                self.metrics.backup_count = len(backups)
                if backups:
                    self.metrics.last_backup = backups[0]['created']
                
            finally:
                self.db_manager.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
    
    def _serialize_for_storage(self, data: Any) -> str:
        """Serialize data for storage with compression."""
        try:
            if data is None:
                return ""
            
            json_str = json.dumps(data, separators=(',', ':'))
            
            # Apply compression if enabled
            if self.compression_manager.compression_type != CompressionType.NONE:
                compressed = self.compression_manager.compress_data(json_str)
                original_size = len(json_str.encode('utf-8'))
                compressed_size = len(compressed)
                
                # Track compression savings
                savings = original_size - compressed_size
                if savings > 0:
                    self.stats["compression_savings"] += savings
                
                # Store as base64 for SQLite compatibility
                import base64
                return base64.b64encode(compressed).decode('ascii')
            
            return json_str
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            return "{}"
    
    def _deserialize_from_storage(self, data: str) -> Any:
        """Deserialize data from storage with decompression."""
        try:
            if not data:
                return {}
            
            # Check if data is compressed (base64 encoded)
            if self.compression_manager.compression_type != CompressionType.NONE:
                try:
                    import base64
                    compressed_data = base64.b64decode(data.encode('ascii'))
                    json_str = self.compression_manager.decompress_data(compressed_data)
                    return json.loads(json_str)
                except Exception:
                    # Fallback to uncompressed data
                    pass
            
            return json.loads(data)
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            return {}
    
    async def get_environmental_data(self, start_time: datetime = None, 
                                   end_time: datetime = None, 
                                   device_id: str = None,
                                   limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve environmental data with filtering."""
        try:
            self.metrics.read_operations += 1
            
            # Build query
            query = "SELECT * FROM environmental_data WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.timestamp())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.timestamp())
            
            if device_id:
                query += " AND device_id = ?"
                params.append(device_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            conn = self.db_manager.get_connection()
            
            try:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in rows:
                    record = dict(zip(columns, row))
                    
                    # Deserialize JSON fields
                    record['sensor_data'] = self._deserialize_from_storage(record['sensor_data'])
                    record['ml_analysis'] = self._deserialize_from_storage(record['ml_analysis'])
                    record['location'] = self._deserialize_from_storage(record['location'])
                    record['metadata'] = self._deserialize_from_storage(record['metadata'])
                    
                    # Convert timestamp
                    record['timestamp'] = datetime.fromtimestamp(record['timestamp'])
                    record['created_at'] = datetime.fromtimestamp(record['created_at'])
                    
                    results.append(record)
                
                return results
                
            finally:
                self.db_manager.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to retrieve environmental data: {e}")
            return []
    
    async def get_alerts(self, start_time: datetime = None, 
                        end_time: datetime = None,
                        severity: str = None,
                        resolved: bool = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve alerts with filtering."""
        try:
            self.metrics.read_operations += 1
            
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.timestamp())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.timestamp())
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            if resolved is not None:
                query += " AND resolved = ?"
                params.append(resolved)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            conn = self.db_manager.get_connection()
            
            try:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in rows:
                    record = dict(zip(columns, row))
                    record['metadata'] = self._deserialize_from_storage(record['metadata'])
                    record['timestamp'] = datetime.fromtimestamp(record['timestamp'])
                    record['created_at'] = datetime.fromtimestamp(record['created_at'])
                    results.append(record)
                
                return results
                
            finally:
                self.db_manager.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to retrieve alerts: {e}")
            return []
    
    async def get_system_events(self, start_time: datetime = None,
                               end_time: datetime = None,
                               event_type: str = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve system events with filtering."""
        try:
            self.metrics.read_operations += 1
            
            query = "SELECT * FROM system_events WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.timestamp())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.timestamp())
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            conn = self.db_manager.get_connection()
            
            try:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in rows:
                    record = dict(zip(columns, row))
                    record['event_data'] = self._deserialize_from_storage(record['event_data'])
                    record['timestamp'] = datetime.fromtimestamp(record['timestamp'])
                    record['created_at'] = datetime.fromtimestamp(record['created_at'])
                    results.append(record)
                
                return results
                
            finally:
                self.db_manager.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to retrieve system events: {e}")
            return []
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        await self._update_metrics()
        
        # Get database file sizes
        db_size = Path(self.db_manager.db_path).stat().st_size if Path(self.db_manager.db_path).exists() else 0
        
        # Get backup information
        backups = self.backup_manager.get_backup_list()
        backup_total_size = sum(b['size'] for b in backups)
        
        return {
            "metrics": {
                "total_records": self.metrics.total_records,
                "total_size_bytes": self.metrics.total_size_bytes,
                "total_size_mb": round(self.metrics.total_size_bytes / 1024 / 1024, 2),
                "records_per_hour": round(self.metrics.records_per_hour, 2),
                "average_record_size": round(self.metrics.average_record_size, 2),
                "compression_ratio": self.metrics.compression_ratio,
                "oldest_record": self.metrics.oldest_record.isoformat() if self.metrics.oldest_record else None,
                "newest_record": self.metrics.newest_record.isoformat() if self.metrics.newest_record else None,
                "write_operations": self.metrics.write_operations,
                "read_operations": self.metrics.read_operations,
                "failed_operations": self.metrics.failed_operations
            },
            "database": {
                "path": str(self.db_path),
                "size_bytes": db_size,
                "size_mb": round(db_size / 1024 / 1024, 2)
            },
            "backups": {
                "count": len(backups),
                "total_size_bytes": backup_total_size,
                "total_size_mb": round(backup_total_size / 1024 / 1024, 2),
                "last_backup": backups[0]['created'].isoformat() if backups else None,
                "list": backups[:5]  # Show only 5 most recent
            },
            "batch_processing": {
                "batch_size": self.batch_size,
                "batch_timeout": self.batch_timeout,
                "pending_records": len(self.write_batch)
            },
            "compression": {
                "type": self.compression_manager.compression_type.value,
                "savings_bytes": self.stats.get("compression_savings", 0),
                "savings_mb": round(self.stats.get("compression_savings", 0) / 1024 / 1024, 2)
            },
            "retention": {
                "retention_days": self.retention_days,
                "cleanup_interval_hours": self.cleanup_interval / 3600,
                "last_cleanup": datetime.fromtimestamp(self.last_cleanup).isoformat() if self.last_cleanup else None
            },
            "operations": {
                "storage_operations": self.stats["storage_operations"],
                "storage_errors": self.stats["storage_errors"],
                "bytes_stored": self.stats["bytes_stored"],
                "error_rate": (
                    self.stats["storage_errors"] / max(self.stats["storage_operations"], 1) * 100
                    if self.stats["storage_operations"] > 0 else 0
                )
            }
        }
    
    async def export_data(self, format: str = "json", 
                         start_time: datetime = None,
                         end_time: datetime = None,
                         include_ml_data: bool = True) -> str:
        """Export data to various formats."""
        try:
            # Get environmental data
            env_data = await self.get_environmental_data(start_time, end_time, limit=10000)
            
            # Get alerts
            alerts = await self.get_alerts(start_time, end_time, limit=1000)
            
            # Get system events
            events = await self.get_system_events(start_time, end_time, limit=1000)
            
            export_data = {
                "export_info": {
                    "timestamp": datetime.now().isoformat(),
                    "device_id": self.device_id,
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": end_time.isoformat() if end_time else None,
                    "format": format
                },
                "environmental_data": env_data,
                "alerts": alerts,
                "system_events": events
            }
            
            if format.lower() == "json":
                return json.dumps(export_data, indent=2, default=str)
            
            elif format.lower() == "csv":
                # Convert to CSV format
                import csv
                import io
                
                output = io.StringIO()
                
                # Environmental data CSV
                if env_data:
                    writer = csv.DictWriter(output, fieldnames=env_data[0].keys())
                    writer.writeheader()
                    writer.writerows(env_data)
                
                return output.getvalue()
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return ""
    
    async def import_data(self, data: str, format: str = "json") -> bool:
        """Import data from external sources."""
        try:
            if format.lower() == "json":
                import_data = json.loads(data)
                
                # Import environmental data
                for record in import_data.get("environmental_data", []):
                    await self.store_environmental_record(record)
                
                # Import alerts
                for alert in import_data.get("alerts", []):
                    await self.store_alert(alert, alert.get("device_id", self.device_id))
                
                logger.info(f"Data import completed successfully")
                return True
            
            else:
                raise ValueError(f"Unsupported import format: {format}")
                
        except Exception as e:
            logger.error(f"Data import failed: {e}")
            return False
    
    async def optimize_database(self):
        """Optimize database performance."""
        try:
            conn = self.db_manager.get_connection()
            
            try:
                # Analyze tables for better query planning
                conn.execute("ANALYZE")
                
                # Vacuum to reclaim space and optimize
                conn.execute("VACUUM")
                
                # Update statistics
                conn.execute("PRAGMA optimize")
                
                logger.info("📦 Database optimization completed")
                
            finally:
                self.db_manager.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
    
    async def shutdown(self):
        """Shutdown storage system gracefully."""
        logger.info("📦 Shutting down storage system...")
        
        try:
            # Cancel batch processing task
            if self.batch_task and not self.batch_task.done():
                self.batch_task.cancel()
                try:
                    await self.batch_task
                except asyncio.CancelledError:
                    pass
            
            # Final batch flush
            async with self.batch_lock:
                if self.write_batch:
                    await self._flush_batch()
            
            # Create final backup
            if self.backup_manager.should_backup():
                await self.backup_manager.create_backup(self.db_manager)
            
            # Close database connections
            self.db_manager.close_all()
            
            logger.info("✅ Storage system shutdown complete")
            
        except Exception as e:
            logger.error(f"Storage shutdown error: {e}")

# Example usage and testing
if __name__ == "__main__":
    async def test_storage_system():
        """Test the storage system."""
        config = {
            "device_id": "test_device",
            "storage_path": "./test_data",
            "batch_size": 10,
            "batch_timeout": 5,
            "compression": "gzip",
            "retention_days": 7,
            "backup_interval": 300  # 5 minutes for testing
        }
        
        storage = StorageManager(config)
        await storage.initialize()
        
        print("Testing storage system...")
        
        # Test environmental data storage
        for i in range(25):
            record = {
                "device_id": "test_device",
                "sensor_data": {
                    "temperature": 20.0 + i,
                    "humidity": 50.0 + i,
                    "pressure": 1013.0 + i
                },
                "ml_analysis": {
                    "anomaly_score": 0.1 + (i * 0.01),
                    "classification": "normal"
                },
                "quality_score": 0.95,
                "location": {"lat": 40.7128, "lon": -74.0060},
                "metadata": {"test": True, "iteration": i}
            }
            await storage.store_environmental_record(record)
            await asyncio.sleep(0.1)
        
        # Test alert storage
        alert = {
            "type": "temperature_high",
            "severity": "warning",
            "message": "Temperature above threshold",
            "metadata": {"threshold": 25.0, "current": 26.5}
        }
        await storage.store_alert(alert, "test_device")
        
        # Test system event storage
        await storage.store_system_event(
            "startup",
            {"version": "1.0.0", "config": "test"},
            "test_device"
        )
        
        # Wait for batch processing
        await asyncio.sleep(6)
        
        # Test data retrieval
        env_data = await storage.get_environmental_data(limit=10)
        print(f"Retrieved {len(env_data)} environmental records")
        
        alerts = await storage.get_alerts(limit=5)
        print(f"Retrieved {len(alerts)} alerts")
        
        events = await storage.get_system_events(limit=5)
        print(f"Retrieved {len(events)} system events")
        
        # Test statistics
        stats = await storage.get_storage_statistics()
        print(f"Storage statistics: {json.dumps(stats, indent=2, default=str)}")
        
        # Test export
        export_data = await storage.export_data(format="json", limit=5)
        print(f"Export data length: {len(export_data)} characters")
        
        await storage.shutdown()
        print("Storage test completed!")
    
    # Run test
    asyncio.run(test_storage_system())