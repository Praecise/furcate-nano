# ============================================================================
# furcate_nano/storage.py
"""Optimized data storage for Furcate Nano devices using DuckDB + RocksDB."""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import struct

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    import rocksdb
    ROCKSDB_AVAILABLE = True
except ImportError:
    ROCKSDB_AVAILABLE = False

# Fallback to sqlite if others unavailable
import sqlite3

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Optimized storage for Raspberry Pi 5 using:
    - DuckDB: Analytics and time-series queries (fast aggregations)
    - RocksDB: High-frequency sensor writes (write-optimized)
    - SQLite: Fallback if others unavailable
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize storage manager."""
        self.config = config
        self.base_path = Path(config.get("data_path", "/data/furcate_nano"))
        self.retention_days = config.get("retention_days", 30)
        self.max_size_mb = config.get("max_size_mb", 1000)
        
        # Database paths
        self.analytics_db = self.base_path / "analytics.duckdb"
        self.timeseries_db = self.base_path / "timeseries"
        self.sqlite_db = self.base_path / "fallback.db"
        
        # Ensure directories exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Database connections
        self.duckdb_conn = None
        self.rocksdb_conn = None
        self.sqlite_conn = None  # Fallback
        
        # Determine which databases to use
        self.use_duckdb = DUCKDB_AVAILABLE and config.get("use_duckdb", True)
        self.use_rocksdb = ROCKSDB_AVAILABLE and config.get("use_rocksdb", True)
        
        # Write batching for performance
        self.write_batch = []
        self.batch_size = config.get("batch_size", 50)
        self.batch_timeout = config.get("batch_timeout", 5.0)  # seconds
        
        # Storage statistics
        self.stats = {
            "records_stored": 0,
            "total_bytes_stored": 0,
            "batch_writes": 0,
            "storage_errors": 0,
            "last_write_time": None,
            "database_sizes": {}
        }
        
        # Background tasks
        self.batch_writer_task = None
        self.maintenance_task = None
        self.shutdown_event = asyncio.Event()
        
        logger.info(f"Storage: DuckDB={self.use_duckdb}, RocksDB={self.use_rocksdb}")
    
    async def initialize(self) -> bool:
        """Initialize storage system."""
        try:
            if self.use_duckdb:
                await self._init_duckdb()
            if self.use_rocksdb:
                await self._init_rocksdb()
            if not self.use_duckdb and not self.use_rocksdb:
                await self._init_sqlite_fallback()
            
            # Start background tasks
            self.batch_writer_task = asyncio.create_task(self._batch_writer_loop())
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())
            
            logger.info("✅ Storage system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Storage initialization failed: {e}")
            return False
    
    async def _init_duckdb(self):
        """Initialize DuckDB for analytics."""
        self.duckdb_conn = duckdb.connect(str(self.analytics_db))
        
        # Create optimized tables for time-series data
        self.duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                timestamp TIMESTAMP,
                device_id VARCHAR,
                sensor_name VARCHAR,
                sensor_type VARCHAR,
                value_json JSON,
                quality DOUBLE,
                confidence DOUBLE,
                metadata JSON
            )
        """)
        
        self.duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_analysis (
                timestamp TIMESTAMP,
                device_id VARCHAR,
                environmental_class VARCHAR,
                anomaly_score DOUBLE,
                confidence DOUBLE,
                features_analyzed INTEGER,
                metadata JSON
            )
        """)
        
        self.duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS environmental_alerts (
                timestamp TIMESTAMP,
                device_id VARCHAR,
                alert_type VARCHAR,
                severity VARCHAR,
                sensor_name VARCHAR,
                parameter_name VARCHAR,
                value DOUBLE,
                threshold_min DOUBLE,
                threshold_max DOUBLE,
                metadata JSON
            )
        """)
        
        self.duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                timestamp TIMESTAMP,
                device_id VARCHAR,
                event_type VARCHAR,
                event_data JSON
            )
        """)
        
        # Create indexes for better query performance
        try:
            self.duckdb_conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_data(timestamp)")
            self.duckdb_conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_device ON sensor_data(device_id)")
            self.duckdb_conn.execute("CREATE INDEX IF NOT EXISTS idx_ml_timestamp ON ml_analysis(timestamp)")
            self.duckdb_conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON environmental_alerts(timestamp)")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
        
        logger.info("✅ DuckDB analytics database ready")
    
    async def _init_rocksdb(self):
        """Initialize RocksDB for high-frequency writes."""
        options = rocksdb.Options()
        options.create_if_missing = True
        options.write_buffer_size = 32 * 1024 * 1024  # 32MB for RPi5
        options.compression = rocksdb.CompressionType.lz4_compression
        options.max_open_files = 100  # Limit open files on RPi
        
        self.rocksdb_conn = rocksdb.DB(str(self.timeseries_db), options)
        logger.info("✅ RocksDB time-series database ready")
    
    async def _init_sqlite_fallback(self):
        """Initialize SQLite as fallback."""
        self.sqlite_conn = sqlite3.connect(str(self.sqlite_db))
        self.sqlite_conn.row_factory = sqlite3.Row
        
        cursor = self.sqlite_conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS environmental_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                device_id TEXT,
                sensor_data TEXT,
                ml_analysis TEXT,
                storage_metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                device_id TEXT,
                alert_data TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_env_timestamp ON environmental_data(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)
        """)
        
        self.sqlite_conn.commit()
        logger.info("✅ SQLite fallback database ready")
    
    def _serialize_for_storage(self, data: Any) -> str:
        """Serialize data for storage, handling special types."""
        def json_serializer(obj):
            """Custom JSON serializer for special types."""
            if hasattr(obj, 'value'):  # Handle Enum types
                return obj.value
            elif hasattr(obj, 'to_dict'):  # Handle custom objects with to_dict method
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):  # Handle other objects
                return obj.__dict__
            else:
                return str(obj)
        
        try:
            return json.dumps(data, default=json_serializer, separators=(',', ':'))
        except Exception as e:
            logger.warning(f"Serialization fallback for {type(data)}: {e}")
            return json.dumps(str(data))
    
    async def store_environmental_record(self, record: Dict[str, Any]) -> bool:
        """Store environmental monitoring record with optimal database selection."""
        try:
            # Add to batch for efficient writing
            storage_record = {
                'type': 'environmental',
                'data': record,
                'timestamp': time.time(),
                'size_bytes': len(self._serialize_for_storage(record))
            }
            
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
        """Store environmental alert."""
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
            
            self.write_batch.append(event_record)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store system event: {e}")
            self.stats["storage_errors"] += 1
            return False
    
    async def _flush_batch(self):
        """Flush write batch to appropriate databases."""
        if not self.write_batch:
            return
        
        try:
            batch = self.write_batch.copy()
            self.write_batch.clear()
            
            for item in batch:
                if item['type'] == 'environmental':
                    await self._write_environmental_data(item['data'])
                elif item['type'] == 'alert':
                    await self._write_alert_data(item['data'])
                elif item['type'] == 'system_event':
                    await self._write_system_event(item['data'])
                
                # Update statistics
                self.stats["records_stored"] += 1
                self.stats["total_bytes_stored"] += item['size_bytes']
            
            self.stats["batch_writes"] += 1
            self.stats["last_write_time"] = time.time()
            
        except Exception as e:
            logger.error(f"Batch flush failed: {e}")
            self.stats["storage_errors"] += 1
    
    async def _write_environmental_data(self, record: Dict[str, Any]):
        """Write environmental data to optimal database."""
        timestamp_str = record.get('timestamp')
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = timestamp_str if isinstance(timestamp_str, datetime) else datetime.now()
        
        device_id = record.get('device_id', 'unknown')
        
        if self.use_duckdb:
            # Write sensor data to DuckDB for analytics
            sensor_data = record.get('sensor_data', {}).get('sensors', {})
            for sensor_name, sensor_info in sensor_data.items():
                try:
                    # Handle both new dict format and old SensorReading format
                    if isinstance(sensor_info, dict):
                        if 'sensor_type' in sensor_info:
                            # New format with sensor_type as string
                            sensor_type = sensor_info['sensor_type']
                            value = sensor_info.get('value', {})
                            quality = sensor_info.get('quality', 1.0)
                            confidence = sensor_info.get('confidence', 1.0)
                            metadata = sensor_info.get('metadata', {})
                        else:
                            # Legacy format
                            sensor_type = sensor_info.get('type', 'unknown')
                            value = sensor_info.get('value', {})
                            quality = sensor_info.get('quality', 1.0)
                            confidence = sensor_info.get('confidence', 1.0)
                            metadata = sensor_info.get('metadata', {})
                    else:
                        # Handle SensorReading objects
                        sensor_type = getattr(sensor_info, 'sensor_type', 'unknown')
                        if hasattr(sensor_type, 'value'):
                            sensor_type = sensor_type.value
                        value = getattr(sensor_info, 'value', {})
                        quality = getattr(sensor_info, 'quality', 1.0)
                        confidence = getattr(sensor_info, 'confidence', 1.0)
                        metadata = getattr(sensor_info, 'metadata', {})
                    
                    self.duckdb_conn.execute("""
                        INSERT INTO sensor_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        device_id,
                        sensor_name,
                        str(sensor_type),
                        self._serialize_for_storage(value),
                        float(quality),
                        float(confidence),
                        self._serialize_for_storage(metadata)
                    ))
                    
                except Exception as e:
                    logger.warning(f"Failed to store sensor data for {sensor_name}: {e}")
            
            # Write ML analysis
            ml_data = record.get('ml_analysis', {})
            if ml_data and not ml_data.get('error'):
                try:
                    self.duckdb_conn.execute("""
                        INSERT INTO ml_analysis VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        device_id,
                        ml_data.get('environmental_class', 'unknown'),
                        float(ml_data.get('anomaly_score', 0.0)),
                        float(ml_data.get('confidence', 0.0)),
                        int(ml_data.get('features_analyzed', 0)),
                        self._serialize_for_storage(ml_data)
                    ))
                except Exception as e:
                    logger.warning(f"Failed to store ML analysis: {e}")
        
        elif self.use_rocksdb:
            # Write to RocksDB with timestamp key
            key = f"{device_id}:{int(timestamp.timestamp())}"
            value = self._serialize_for_storage(record)
            self.rocksdb_conn.put(key.encode('utf-8'), value.encode('utf-8'))
        
        else:
            # Fallback to SQLite
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO environmental_data (timestamp, device_id, sensor_data, ml_analysis, storage_metadata) 
                VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp,
                device_id,
                self._serialize_for_storage(record.get('sensor_data', {})),
                self._serialize_for_storage(record.get('ml_analysis', {})),
                self._serialize_for_storage({'cycle': record.get('cycle', 0)})
            ))
            self.sqlite_conn.commit()
    
    async def _write_alert_data(self, alert_data: Dict[str, Any]):
        """Write alert data to database."""
        timestamp_str = alert_data.get('timestamp')
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = datetime.now()
        
        device_id = alert_data.get('device_id', 'unknown')
        alert = alert_data.get('alert', {})
        
        if self.use_duckdb:
            try:
                self.duckdb_conn.execute("""
                    INSERT INTO environmental_alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    device_id,
                    alert.get('type', 'unknown'),
                    alert.get('severity', 'warning'),
                    alert.get('sensor', ''),
                    alert.get('parameter', ''),
                    float(alert.get('value', 0.0)) if alert.get('value') is not None else None,
                    float(alert.get('threshold', [0, 0])[0]) if alert.get('threshold') else None,
                    float(alert.get('threshold', [0, 0])[1]) if alert.get('threshold') and len(alert.get('threshold', [])) > 1 else None,
                    self._serialize_for_storage(alert)
                ))
            except Exception as e:
                logger.warning(f"Failed to store alert to DuckDB: {e}")
        
        elif self.use_rocksdb:
            key = f"alert:{device_id}:{int(timestamp.timestamp())}"
            value = self._serialize_for_storage(alert_data)
            self.rocksdb_conn.put(key.encode('utf-8'), value.encode('utf-8'))
        
        else:
            # SQLite fallback
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (timestamp, device_id, alert_data) VALUES (?, ?, ?)
            """, (
                timestamp,
                device_id,
                self._serialize_for_storage(alert_data)
            ))
            self.sqlite_conn.commit()
    
    async def _write_system_event(self, event_data: Dict[str, Any]):
        """Write system event to database."""
        timestamp_str = event_data.get('timestamp')
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = datetime.now()
        
        device_id = event_data.get('device_id', 'unknown')
        event_type = event_data.get('event_type', 'unknown')
        event_info = event_data.get('event_data', {})
        
        if self.use_duckdb:
            try:
                self.duckdb_conn.execute("""
                    INSERT INTO system_events VALUES (?, ?, ?, ?)
                """, (
                    timestamp,
                    device_id,
                    event_type,
                    self._serialize_for_storage(event_info)
                ))
            except Exception as e:
                logger.warning(f"Failed to store system event: {e}")
    
    async def get_recent_environmental_data(self, hours: int = 24, device_id: str = None) -> List[Dict[str, Any]]:
        """Get recent environmental data using optimal query method."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            if self.use_duckdb:
                # Use DuckDB for fast analytics
                query = """
                    SELECT 
                        timestamp,
                        device_id,
                        sensor_name,
                        sensor_type,
                        value_json,
                        quality,
                        confidence,
                        metadata
                    FROM sensor_data 
                    WHERE timestamp > ?
                """
                params = [cutoff_time]
                
                if device_id:
                    query += " AND device_id = ?"
                    params.append(device_id)
                
                query += " ORDER BY timestamp DESC LIMIT 1000"
                
                result = self.duckdb_conn.execute(query, params).fetchall()
                
                # Convert to list of dictionaries
                records = []
                for row in result:
                    record = {
                        'timestamp': row[0],
                        'device_id': row[1],
                        'sensor_name': row[2],
                        'sensor_type': row[3],
                        'value': json.loads(row[4]) if row[4] else {},
                        'quality': row[5],
                        'confidence': row[6],
                        'metadata': json.loads(row[7]) if row[7] else {}
                    }
                    records.append(record)
                
                return records
            
            elif self.use_rocksdb:
                # Scan RocksDB (less efficient for range queries but works)
                records = []
                it = self.rocksdb_conn.iterkeys()
                it.seek_to_first()
                
                for key in it:
                    try:
                        key_str = key.decode('utf-8')
                        if device_id and not key_str.startswith(f"{device_id}:"):
                            continue
                        
                        value = self.rocksdb_conn.get(key)
                        record = json.loads(value.decode('utf-8'))
                        
                        # Check timestamp
                        record_time_str = record.get('timestamp', '')
                        if record_time_str:
                            record_time = datetime.fromisoformat(record_time_str.replace('Z', '+00:00'))
                            if record_time > cutoff_time:
                                records.append(record)
                    except Exception as e:
                        logger.debug(f"Skipping corrupted record: {e}")
                        continue
                
                return sorted(records, key=lambda x: x.get('timestamp', ''), reverse=True)[:1000]
            
            else:
                # SQLite fallback
                cursor = self.sqlite_conn.cursor()
                query = """
                    SELECT * FROM environmental_data 
                    WHERE timestamp > ? 
                """
                params = [cutoff_time]
                
                if device_id:
                    query += " AND device_id = ? "
                    params.append(device_id)
                
                query += " ORDER BY timestamp DESC LIMIT 1000"
                
                cursor.execute(query, params)
                
                records = []
                for row in cursor.fetchall():
                    record = {
                        'id': row['id'],
                        'timestamp': row['timestamp'],
                        'device_id': row['device_id'],
                        'sensor_data': json.loads(row['sensor_data']) if row['sensor_data'] else {},
                        'ml_analysis': json.loads(row['ml_analysis']) if row['ml_analysis'] else {},
                        'storage_metadata': json.loads(row['storage_metadata']) if row['storage_metadata'] else {}
                    }
                    records.append(record)
                
                return records
            
        except Exception as e:
            logger.error(f"Failed to get environmental data: {e}")
            return []
    
    async def get_alerts(self, hours: int = 24, device_id: str = None, severity: str = None) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            if self.use_duckdb:
                query = """
                    SELECT * FROM environmental_alerts 
                    WHERE timestamp > ?
                """
                params = [cutoff_time]
                
                if device_id:
                    query += " AND device_id = ?"
                    params.append(device_id)
                
                if severity:
                    query += " AND severity = ?"
                    params.append(severity)
                
                query += " ORDER BY timestamp DESC"
                
                result = self.duckdb_conn.execute(query, params).fetchall()
                return [dict(zip(['timestamp', 'device_id', 'alert_type', 'severity', 'sensor_name', 'parameter_name', 'value', 'threshold_min', 'threshold_max', 'metadata'], row)) for row in result]
            
            else:
                # SQLite fallback
                cursor = self.sqlite_conn.cursor()
                query = "SELECT * FROM alerts WHERE timestamp > ?"
                params = [cutoff_time]
                
                if device_id:
                    query += " AND device_id = ?"
                    params.append(device_id)
                
                query += " ORDER BY timestamp DESC"
                cursor.execute(query, params)
                
                return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    async def _batch_writer_loop(self):
        """Background batch writer for performance."""
        while not self.shutdown_event.is_set():
            try:
                # Wait for timeout or shutdown
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=self.batch_timeout)
                break  # Shutdown requested
            except asyncio.TimeoutError:
                # Timeout - flush batch if it has data
                if self.write_batch:
                    await self._flush_batch()
    
    async def _maintenance_loop(self):
        """Enhanced maintenance for multiple databases."""
        while not self.shutdown_event.is_set():
            try:
                # Wait for maintenance interval or shutdown
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=3600)  # 1 hour
                break  # Shutdown requested
            except asyncio.TimeoutError:
                # Run maintenance
                await self._run_maintenance()
    
    async def _run_maintenance(self):
        """Run database maintenance tasks."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            
            # Clean old data from DuckDB
            if self.use_duckdb and self.duckdb_conn:
                deleted_sensors = self.duckdb_conn.execute("DELETE FROM sensor_data WHERE timestamp < ?", (cutoff_time,)).fetchone()
                deleted_ml = self.duckdb_conn.execute("DELETE FROM ml_analysis WHERE timestamp < ?", (cutoff_time,)).fetchone()
                deleted_alerts = self.duckdb_conn.execute("DELETE FROM environmental_alerts WHERE timestamp < ?", (cutoff_time,)).fetchone()
                deleted_events = self.duckdb_conn.execute("DELETE FROM system_events WHERE timestamp < ?", (cutoff_time,)).fetchone()
                
                # Optimize database
                self.duckdb_conn.execute("VACUUM")
                
                logger.info(f"DuckDB maintenance: cleaned old records")
            
            # Clean old data from SQLite
            if self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("DELETE FROM environmental_data WHERE timestamp < ?", (cutoff_time,))
                cursor.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff_time,))
                cursor.execute("VACUUM")
                self.sqlite_conn.commit()
                
                logger.info("SQLite maintenance: cleaned old records")
            
            # Update database size statistics
            await self._update_size_stats()
            
            logger.debug("Storage maintenance completed")
            
        except Exception as e:
            logger.error(f"Storage maintenance error: {e}")
    
    async def _update_size_stats(self):
        """Update database size statistics."""
        try:
            sizes = {}
            
            if self.analytics_db.exists():
                sizes['duckdb_mb'] = self.analytics_db.stat().st_size / (1024 * 1024)
            
            if self.timeseries_db.exists():
                total_size = sum(f.stat().st_size for f in self.timeseries_db.rglob('*') if f.is_file())
                sizes['rocksdb_mb'] = total_size / (1024 * 1024)
            
            if self.sqlite_db.exists():
                sizes['sqlite_mb'] = self.sqlite_db.stat().st_size / (1024 * 1024)
            
            self.stats['database_sizes'] = sizes
            
        except Exception as e:
            logger.warning(f"Failed to update size stats: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            **self.stats,
            'write_batch_size': len(self.write_batch),
            'storage_config': {
                'use_duckdb': self.use_duckdb,
                'use_rocksdb': self.use_rocksdb,
                'retention_days': self.retention_days,
                'batch_size': self.batch_size
            }
        }
    
    async def export_data(self, start_time: datetime, end_time: datetime, format: str = 'json') -> str:
        """Export data for backup or analysis."""
        try:
            data = await self.get_recent_environmental_data(
                hours=int((end_time - start_time).total_seconds() / 3600)
            )
            
            if format == 'json':
                return json.dumps(data, indent=2, default=str)
            elif format == 'csv':
                # Convert to CSV format
                import csv
                import io
                
                output = io.StringIO()
                if data:
                    writer = csv.DictWriter(output, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return ""
    
    async def shutdown(self):
        """Shutdown storage system."""
        logger.info("💾 Shutting down storage manager...")
        
        try:
            # Signal shutdown to background tasks
            self.shutdown_event.set()
            
            # Wait for background tasks to complete
            if self.batch_writer_task:
                await self.batch_writer_task
            if self.maintenance_task:
                await self.maintenance_task
            
            # Flush any remaining data
            await self._flush_batch()
            
            # Close database connections
            if self.duckdb_conn:
                self.duckdb_conn.close()
                logger.info("✅ DuckDB connection closed")
            
            if self.rocksdb_conn:
                del self.rocksdb_conn  # RocksDB doesn't have explicit close
                logger.info("✅ RocksDB connection closed")
            
            if self.sqlite_conn:
                self.sqlite_conn.close()
                logger.info("✅ SQLite connection closed")
            
        except Exception as e:
            logger.error(f"Storage shutdown error: {e}")
        
        logger.info("✅ Storage manager shutdown complete")