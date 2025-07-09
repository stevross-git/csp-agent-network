"""DNS overlay implementation for .web4ai and peoplesainetwork.com domains."""
import asyncio
import logging
import json
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import hashlib


class DNSRecord:
    """DNS record entry."""
    def __init__(self, name: str, value: str, ttl: int = 3600, record_type: str = "A"):
        self.name = name
        self.value = value
        self.ttl = ttl
        self.record_type = record_type
        self.created_at = datetime.utcnow()
        
    def is_expired(self) -> bool:
        """Check if record is expired."""
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for DHT storage."""
        return {
            "name": self.name,
            "value": self.value,
            "ttl": self.ttl,
            "type": self.record_type,
            "created": self.created_at.timestamp()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DNSRecord':
        """Create record from dictionary."""
        record = cls(
            data["name"],
            data["value"],
            data.get("ttl", 3600),
            data.get("type", "A")
        )
        if "created" in data:
            record.created_at = datetime.fromtimestamp(data["created"])
        return record


class DNSOverlay:
    """DNS overlay for .web4ai and peoplesainetwork.com domain resolution."""
    
    def __init__(self, network_node):
        self.node = network_node
        self.records: Dict[str, DNSRecord] = {}
        self.cache: Dict[str, DNSRecord] = {}
        self.logger = logging.getLogger("enhanced_csp.dns")
        self.is_running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # DNS query retry settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.query_timeout = 5.0
        
        # DHT key prefix for DNS records
        self.dns_key_prefix = "dns:"
        
        # Supported domain suffixes (order matters - first is default)
        self.supported_domains = ['.web4ai', '.peoplesainetwork.com']
        
    async def start(self):
        """Start DNS overlay service."""
        if self.is_running:
            return
            
        self.logger.info("Starting DNS overlay")
        self.is_running = True
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
        
    async def stop(self):
        """Stop DNS overlay service."""
        if not self.is_running:
            return
            
        self.logger.info("Stopping DNS overlay")
        self.is_running = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
    async def register(self, name: str, value: str, ttl: int = 3600, record_type: str = "A") -> bool:
        """Register a DNS name."""
        if not self._validate_domain_name(name):
            self.logger.error(f"Invalid domain name: {name}")
            return False
            
        # Handle both .web4ai and peoplesainetwork.com domains
        original_name = name
        if not self._has_supported_domain_suffix(name):
            # If no domain suffix, default to .web4ai for backward compatibility
            name = f"{name}.web4ai"
            
        record = DNSRecord(name, value, ttl, record_type)
        self.records[name] = record
        self.logger.info(f"Registered DNS: {name} -> {value} (type: {record_type})")
        
        # Propagate to network if not genesis or if we have DHT
        if hasattr(self.node, 'dht') and self.node.dht:
            success = await self._propagate_record(record)
            if not success:
                self.logger.warning(f"Failed to propagate DNS record: {name}")
                return False
                
        return True
        
    async def resolve(self, name: str, record_type: str = "A") -> Optional[str]:
        """Resolve a DNS name with fallback support for multiple domains."""
        if not self._validate_domain_name(name):
            return None
        
        # Try exact match first
        if self._has_supported_domain_suffix(name):
            result = await self._resolve_exact(name, record_type)
            if result:
                return result
        else:
            # If not found and no domain suffix, try both supported domains
            # Try .web4ai first (backward compatibility)
            web4ai_name = f"{name}.web4ai"
            result = await self._resolve_exact(web4ai_name, record_type)
            if result:
                return result
            
            # Try .peoplesainetwork.com
            peoples_name = f"{name}.peoplesainetwork.com"
            result = await self._resolve_exact(peoples_name, record_type)
            if result:
                return result
        
        return None
    
    async def _resolve_exact(self, name: str, record_type: str = "A") -> Optional[str]:
        """Resolve exact domain name."""
        cache_key = f"{name}:{record_type}"
        
        # Check local records first
        if name in self.records and not self.records[name].is_expired():
            record = self.records[name]
            if record.record_type == record_type:
                return record.value
            
        # Check cache
        if cache_key in self.cache and not self.cache[cache_key].is_expired():
            return self.cache[cache_key].value
        
        # Try DHT lookup if available
        if hasattr(self.node, 'dht') and self.node.dht:
            result = await self._resolve_from_dht(name, record_type)
            if result:
                # Cache the result
                record = DNSRecord(name, result, record_type=record_type)
                self.cache[cache_key] = record
                self.logger.debug(f"Resolved {name} -> {result}")
                return result
        
        # Fallback to upstream DNS for external domains
        if name.endswith('.peoplesainetwork.com'):
            result = await self._resolve_upstream(name, record_type)
            if result:
                # Cache the result
                record = DNSRecord(name, result, ttl=300, record_type=record_type)  # Shorter TTL for external
                self.cache[cache_key] = record
                return result
        
        return None
    
    async def _resolve_from_dht(self, name: str, record_type: str = "A") -> Optional[str]:
        """Query the network for a DNS name via DHT."""
        dht_key = self._get_dht_key(name)
        
        # Query network with retries
        for attempt in range(self.max_retries):
            try:
                result = await asyncio.wait_for(
                    self._query_network(name, record_type),
                    timeout=self.query_timeout
                )
                
                if result:
                    self.logger.debug(f"Resolved {name} from DHT -> {result}")
                    return result
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"DNS query timeout for {name} (attempt {attempt + 1})")
            except Exception as e:
                self.logger.error(f"DNS query error for {name}: {e}")
                
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
        return None
    
    async def _resolve_upstream(self, name: str, record_type: str = "A") -> Optional[str]:
        """Resolve using upstream DNS servers."""
        try:
            import aiohttp
            import socket
            
            # Try to resolve using system DNS
            if record_type == "A":
                try:
                    result = socket.gethostbyname(name)
                    self.logger.debug(f"Resolved {name} via system DNS -> {result}")
                    return result
                except socket.gaierror:
                    pass
            
            # TODO: Implement proper DNS over HTTPS or DNS over TLS
            # For now, just return None for non-A records
            
        except Exception as e:
            self.logger.error(f"Upstream DNS resolution failed for {name}: {e}")
        
        return None
        
    async def list_records(self) -> Dict[str, Dict[str, Any]]:
        """List all DNS records."""
        result = {}
        for name, record in self.records.items():
            if not record.is_expired():
                result[name] = {
                    "value": record.value,
                    "type": record.record_type,
                    "ttl": record.ttl,
                    "created": record.created_at.isoformat()
                }
        return result
        
    async def delete_record(self, name: str) -> bool:
        """Delete a DNS record."""
        if not self._has_supported_domain_suffix(name):
            # Try both domain suffixes for deletion
            web4ai_name = f"{name}.web4ai"
            peoples_name = f"{name}.peoplesainetwork.com"
            
            deleted = False
            if web4ai_name in self.records:
                del self.records[web4ai_name]
                self.logger.info(f"Deleted DNS record: {web4ai_name}")
                deleted = True
                
                # Remove from DHT if available
                if hasattr(self.node, 'dht') and self.node.dht:
                    await self._propagate_deletion(web4ai_name)
                    
            if peoples_name in self.records:
                del self.records[peoples_name]
                self.logger.info(f"Deleted DNS record: {peoples_name}")
                deleted = True
                
                # Remove from DHT if available
                if hasattr(self.node, 'dht') and self.node.dht:
                    await self._propagate_deletion(peoples_name)
                    
            return deleted
        else:
            # Delete specific domain
            if name in self.records:
                del self.records[name]
                self.logger.info(f"Deleted DNS record: {name}")
                
                # Remove from DHT if available
                if hasattr(self.node, 'dht') and self.node.dht:
                    await self._propagate_deletion(name)
                    
                return True
        return False
    
    async def _propagate_deletion(self, name: str):
        """Propagate DNS record deletion to the network."""
        dht_key = self._get_dht_key(name)
        try:
            # Store an empty/expired record to indicate deletion
            await self.node.dht.store(dht_key, {"deleted": True}, ttl=1)
        except Exception as e:
            self.logger.error(f"Failed to propagate DNS deletion for {name}: {e}")
        
    async def bulk_register(self, records: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Register multiple DNS records at once."""
        results = {}
        for record_data in records:
            name = record_data.get("name")
            value = record_data.get("value")
            ttl = record_data.get("ttl", 3600)
            record_type = record_data.get("type", "A")
            
            if name and value:
                results[name] = await self.register(name, value, ttl, record_type)
            else:
                results[name] = False
                
        return results
    
    def _has_supported_domain_suffix(self, name: str) -> bool:
        """Check if name has a supported domain suffix."""
        return any(name.endswith(domain) for domain in self.supported_domains)
        
    def _validate_domain_name(self, name: str) -> bool:
        """Validate domain name format."""
        if not name or len(name) > 253:
            return False
            
        # Allow temporary names without domain suffix for internal use
        if not self._has_supported_domain_suffix(name):
            # For registration, we'll add appropriate domain suffix
            return len(name) > 0 and (name.count('.') <= 2)
            
        # Validate supported domain formats
        if name.endswith('.web4ai'):
            # Basic validation for .web4ai domains
            parts = name[:-8].split('.')  # Remove .web4ai suffix
            return all(
                len(part) > 0 and len(part) <= 63 and
                part.replace('-', '').replace('_', '').isalnum()
                for part in parts
            )
        elif name.endswith('.peoplesainetwork.com'):
            # Basic validation for .peoplesainetwork.com domains
            parts = name[:-21].split('.')  # Remove .peoplesainetwork.com suffix
            return all(
                len(part) > 0 and len(part) <= 63 and
                part.replace('-', '').replace('_', '').isalnum()
                for part in parts
            )
            
        return False
        
    def _get_dht_key(self, name: str) -> str:
        """Get DHT key for DNS name."""
        # Use SHA256 hash for consistent key distribution
        key_data = f"{self.dns_key_prefix}{name}".encode()
        return hashlib.sha256(key_data).hexdigest()
        
    async def _propagate_record(self, record: DNSRecord) -> bool:
        """Propagate DNS record to the network via DHT."""
        if not hasattr(self.node, 'dht') or not self.node.dht:
            self.logger.warning("DHT not available for DNS propagation")
            return False
            
        dht_key = self._get_dht_key(record.name)
        record_data = record.to_dict()
        
        try:
            success = await self.node.dht.store(dht_key, record_data, ttl=record.ttl)
            if success:
                self.logger.debug(f"Propagated DNS record {record.name} to DHT")
            else:
                self.logger.warning(f"Failed to store DNS record {record.name} in DHT")
            return success
        except Exception as e:
            self.logger.error(f"Error propagating DNS record {record.name}: {e}")
            return False
        
    async def _query_network(self, name: str, record_type: str = "A") -> Optional[str]:
        """Query the network for a DNS name via DHT."""
        if not hasattr(self.node, 'dht') or not self.node.dht:
            return None
            
        dht_key = self._get_dht_key(name)
        
        try:
            result = await self.node.dht.get(dht_key)
            if result:
                # Handle different result formats
                if isinstance(result, dict):
                    # Check if record was deleted
                    if result.get("deleted"):
                        return None
                        
                    # Validate record data
                    if (result.get("name") == name and 
                        result.get("type", "A") == record_type):
                        
                        # Check if record is still valid
                        if "created" in result:
                            created_time = datetime.fromtimestamp(result["created"])
                            ttl = result.get("ttl", 3600)
                            if datetime.utcnow() > created_time + timedelta(seconds=ttl):
                                self.logger.debug(f"DNS record {name} expired in DHT")
                                return None
                                
                        return result.get("value")
                elif isinstance(result, str):
                    # Legacy format - just return the value
                    return result
                    
        except Exception as e:
            self.logger.error(f"Error querying DHT for {name}: {e}")
            
        return None
        
    async def _cleanup_expired(self):
        """Cleanup expired records periodically."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = datetime.utcnow()
                
                # Clean expired local records
                expired_records = [
                    name for name, record in self.records.items()
                    if record.is_expired()
                ]
                for name in expired_records:
                    del self.records[name]
                    self.logger.debug(f"Cleaned expired DNS record: {name}")
                
                # Clean expired cache entries
                expired_cache = [
                    name for name, record in self.cache.items()
                    if record.is_expired()
                ]
                for name in expired_cache:
                    del self.cache[name]
                    self.logger.debug(f"Cleaned expired DNS cache: {name}")
                    
                # Log cleanup stats if we cleaned anything
                if expired_records or expired_cache:
                    self.logger.info(
                        f"DNS cleanup: removed {len(expired_records)} records, "
                        f"{len(expired_cache)} cache entries"
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in DNS cleanup task: {e}")
                
    async def get_stats(self) -> Dict[str, Any]:
        """Get DNS overlay statistics."""
        valid_records = sum(1 for r in self.records.values() if not r.is_expired())
        valid_cache = sum(1 for r in self.cache.values() if not r.is_expired())
        
        # Count records by domain
        web4ai_records = sum(1 for name in self.records.keys() if name.endswith('.web4ai'))
        peoples_records = sum(1 for name in self.records.keys() if name.endswith('.peoplesainetwork.com'))
        
        return {
            "total_records": len(self.records),
            "valid_records": valid_records,
            "cached_records": len(self.cache),
            "valid_cached": valid_cache,
            "web4ai_records": web4ai_records,
            "peoplesainetwork_records": peoples_records,
            "supported_domains": self.supported_domains,
            "is_running": self.is_running,
            "has_dht": hasattr(self.node, 'dht') and self.node.dht is not None
        }
        
    async def refresh_record(self, name: str) -> bool:
        """Refresh a DNS record by re-propagating it to the network."""
        if not self._has_supported_domain_suffix(name):
            # Try to refresh both domain variants
            web4ai_name = f"{name}.web4ai"
            peoples_name = f"{name}.peoplesainetwork.com"
            
            refreshed = False
            if web4ai_name in self.records:
                record = self.records[web4ai_name]
                if not record.is_expired():
                    record.created_at = datetime.utcnow()
                    refreshed = await self._propagate_record(record) or refreshed
                    
            if peoples_name in self.records:
                record = self.records[peoples_name]
                if not record.is_expired():
                    record.created_at = datetime.utcnow()
                    refreshed = await self._propagate_record(record) or refreshed
                    
            return refreshed
        else:
            # Refresh specific domain
            if name in self.records:
                record = self.records[name]
                if not record.is_expired():
                    # Update created time and re-propagate
                    record.created_at = datetime.utcnow()
                    return await self._propagate_record(record)
        return False