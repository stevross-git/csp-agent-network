# enhanced_csp/network/dns/overlay.py
"""
DNS overlay implementation for .web4ai domain
Provides decentralized DNS with DHT backend and DNSSEC signing
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import struct

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.backends import default_backend

from ..core.types import NodeID, DNSRecord, DNSConfig
from ..core.node import NetworkNode
from ..p2p.dht import KademliaDHT


logger = logging.getLogger(__name__)


class DNSRecordType(Enum):
    """DNS record types"""
    A = "A"          # IPv4 address
    AAAA = "AAAA"    # IPv6 address
    TXT = "TXT"      # Text record
    SRV = "SRV"      # Service record
    CNAME = "CNAME"  # Canonical name
    MX = "MX"        # Mail exchange
    NS = "NS"        # Name server
    KEY = "KEY"      # Public key (for DNSSEC)


@dataclass
class DNSQuery:
    """DNS query structure"""
    name: str
    record_type: DNSRecordType
    query_id: int
    flags: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class DNSResponse:
    """DNS response structure"""
    query: DNSQuery
    records: List[DNSRecord]
    authoritative: bool = False
    additional: List[DNSRecord] = field(default_factory=list)
    rcode: int = 0  # Response code (0 = success)


@dataclass
class DNSZone:
    """DNS zone for authoritative records"""
    name: str  # e.g., "alice.web4ai"
    owner_key: ed25519.Ed25519PublicKey
    records: List[DNSRecord] = field(default_factory=list)
    serial: int = 0
    refresh: int = 3600
    retry: int = 600
    expire: int = 86400
    minimum: int = 300
    signatures: Dict[str, bytes] = field(default_factory=dict)


class DNSOverlay:
    """DNS overlay network for .web4ai domains"""
    
    ROOT_DOMAIN = ".web4ai"
    CACHE_SIZE = 10000
    
    def __init__(self, node: NetworkNode, config: DNSConfig):
        self.node = node
        self.config = config
        self.dht: Optional[KademliaDHT] = None
        
        # Local zones we're authoritative for
        self.authoritative_zones: Dict[str, DNSZone] = {}
        
        # DNS cache
        self.cache: Dict[Tuple[str, DNSRecordType], DNSResponse] = {}
        self.cache_order: List[Tuple[str, DNSRecordType]] = []
        
        # Resolver state
        self.pending_queries: Dict[int, asyncio.Future] = {}
        self.query_id_counter = 0
        
        # DNSSEC keys
        self.zone_signing_key = self.node.private_key
        self.zone_signing_public = self.node.public_key
        
        # Tasks
        self._tasks: List[asyncio.Task] = []
    
    async def start(self, dht: KademliaDHT):
        """Start DNS overlay"""
        self.dht = dht
        logger.info(f"Starting DNS overlay for {self.ROOT_DOMAIN}")
        
        # Register our node name
        await self._register_node_name()
        
        # Start maintenance tasks
        self._tasks.extend([
            asyncio.create_task(self._cache_cleanup_loop()),
            asyncio.create_task(self._zone_refresh_loop())
        ])
        
        # Register message handlers
        self.node.on_event('dns_query', self.handle_dns_query)
        self.node.on_event('dns_response', self.handle_dns_response)
    
    async def stop(self):
        """Stop DNS overlay"""
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
    
    async def _register_node_name(self):
        """Register our node's DNS name"""
        # Use first 8 chars of node ID as subdomain
        node_name = f"{self.node.node_id.to_base58()[:8]}{self.ROOT_DOMAIN}"
        
        # Create A record for our node
        a_record = DNSRecord(
            name=node_name,
            record_type="A",
            value=self._get_our_ip(),
            ttl=3600
        )
        
        # Create KEY record for DNSSEC
        key_record = DNSRecord(
            name=node_name,
            record_type="KEY",
            value=self._encode_public_key(self.zone_signing_public),
            ttl=86400
        )
        
        # Create zone
        zone = DNSZone(
            name=node_name,
            owner_key=self.zone_signing_public,
            records=[a_record, key_record]
        )
        
        # Sign zone
        self._sign_zone(zone)
        
        # Store as authoritative
        self.authoritative_zones[node_name] = zone
        
        # Publish to DHT
        await self._publish_zone(zone)
        
        logger.info(f"Registered DNS name: {node_name}")
    
    def _get_our_ip(self) -> str:
        """Get our external IP address"""
        # In production, would use NAT detection
        # For now, return placeholder
        if hasattr(self.node, 'nat_info') and self.node.nat_info:
            return self.node.nat_info.external_ip
        return "0.0.0.0"
    
    def _encode_public_key(self, key: ed25519.Ed25519PublicKey) -> str:
        """Encode public key for DNS KEY record"""
        key_bytes = key.public_bytes_raw()
        # Format: flags(2) protocol(1) algorithm(1) public_key
        # Flags: 256 = Zone key
        # Protocol: 3 = DNSSEC
        # Algorithm: 15 = Ed25519
        encoded = struct.pack("!HBB", 256, 3, 15) + key_bytes
        return encoded.hex()
    
    def _sign_zone(self, zone: DNSZone):
        """Sign all records in zone with DNSSEC"""
        for record in zone.records:
            # Create signature
            signature = self._sign_record(record, zone.name)
            zone.signatures[self._record_key(record)] = signature
            record.signature = signature
    
    def _sign_record(self, record: DNSRecord, zone_name: str) -> bytes:
        """Sign a DNS record"""
        # Create data to sign
        data = (
            f"{record.name}|{record.record_type}|{record.value}|"
            f"{record.ttl}|{zone_name}"
        ).encode()
        
        # Sign with Ed25519
        signature = self.zone_signing_key.sign(data)
        return signature
    
    def _verify_record(self, record: DNSRecord, zone_name: str,
                      public_key: ed25519.Ed25519PublicKey) -> bool:
        """Verify DNS record signature"""
        if not record.signature:
            return False
        
        try:
            # Create data that was signed
            data = (
                f"{record.name}|{record.record_type}|{record.value}|"
                f"{record.ttl}|{zone_name}"
            ).encode()
            
            # Verify signature
            public_key.verify(record.signature, data)
            return True
        except Exception:
            return False
    
    def _record_key(self, record: DNSRecord) -> str:
        """Generate key for record"""
        return f"{record.name}:{record.record_type}"
    
    async def _publish_zone(self, zone: DNSZone):
        """Publish zone to DHT"""
        if not self.dht:
            return
        
        # Create zone key for DHT
        zone_key = self._get_zone_key(zone.name)
        
        # Serialize zone
        zone_data = {
            'name': zone.name,
            'owner_key': self._encode_public_key(zone.owner_key),
            'serial': zone.serial,
            'records': [
                {
                    'name': r.name,
                    'type': r.record_type,
                    'value': r.value,
                    'ttl': r.ttl,
                    'signature': r.signature.hex() if r.signature else None
                }
                for r in zone.records
            ],
            'updated': time.time()
        }
        
        # Publish to DHT
        await self.dht.put(zone_key, json.dumps(zone_data).encode())
        
        # Also publish individual records for faster lookup
        for record in zone.records:
            record_key = self._get_record_key(record.name, record.record_type)
            await self.dht.put(record_key, json.dumps({
                'zone': zone.name,
                'record': {
                    'name': record.name,
                    'type': record.record_type,
                    'value': record.value,
                    'ttl': record.ttl,
                    'signature': record.signature.hex() if record.signature else None
                }
            }).encode())
    
    def _get_zone_key(self, zone_name: str) -> bytes:
        """Generate DHT key for zone"""
        return hashlib.sha256(f"dns:zone:{zone_name}".encode()).digest()
    
    def _get_record_key(self, name: str, record_type: str) -> bytes:
        """Generate DHT key for record"""
        return hashlib.sha256(f"dns:record:{name}:{record_type}".encode()).digest()
    
    async def resolve(self, name: str, record_type: DNSRecordType,
                     timeout: float = 5.0) -> Optional[DNSResponse]:
        """Resolve DNS name"""
        # Check if it's a .web4ai domain
        if not name.endswith(self.ROOT_DOMAIN):
            logger.warning(f"Cannot resolve non-web4ai domain: {name}")
            return None
        
        # Check cache first
        cache_key = (name, record_type)
        if cache_key in self.cache:
            response = self.cache[cache_key]
            if not any(r.is_expired() for r in response.records):
                return response
        
        # Check if we're authoritative
        if name in self.authoritative_zones:
            return self._resolve_authoritative(name, record_type)
        
        # Query DHT
        response = await self._query_dht(name, record_type)
        if response:
            self._cache_response(response)
            return response
        
        # Query other nodes
        return await self._query_nodes(name, record_type, timeout)
    
    def _resolve_authoritative(self, name: str, 
                             record_type: DNSRecordType) -> DNSResponse:
        """Resolve from authoritative zone"""
        zone = self.authoritative_zones[name]
        
        # Find matching records
        records = [
            r for r in zone.records
            if r.name == name and r.record_type == record_type.value
        ]
        
        # Create response
        query = DNSQuery(name, record_type, 0)
        return DNSResponse(
            query=query,
            records=records,
            authoritative=True
        )
    
    async def _query_dht(self, name: str, 
                        record_type: DNSRecordType) -> Optional[DNSResponse]:
        """Query DHT for DNS record"""
        if not self.dht:
            return None
        
        try:
            # Look up record in DHT
            record_key = self._get_record_key(name, record_type.value)
            data = await self.dht.get(record_key)
            
            if data:
                record_data = json.loads(data.decode())
                record_info = record_data['record']
                
                # Create DNSRecord
                record = DNSRecord(
                    name=record_info['name'],
                    record_type=record_info['type'],
                    value=record_info['value'],
                    ttl=record_info['ttl'],
                    signature=bytes.fromhex(record_info['signature']) 
                    if record_info.get('signature') else None
                )
                
                # Verify signature if DNSSEC enabled
                if self.config.enable_dnssec and record.signature:
                    # Look up zone to get public key
                    zone_name = record_data.get('zone', name)
                    zone_key = self._get_zone_key(zone_name)
                    zone_data = await self.dht.get(zone_key)
                    
                    if zone_data:
                        zone_info = json.loads(zone_data.decode())
                        # Verify record
                        # TODO: Decode public key and verify
                
                # Create response
                query = DNSQuery(name, record_type, 0)
                return DNSResponse(
                    query=query,
                    records=[record],
                    authoritative=False
                )
            
        except Exception as e:
            logger.error(f"DHT query failed for {name}: {e}")
        
        return None
    
    async def _query_nodes(self, name: str, record_type: DNSRecordType,
                          timeout: float) -> Optional[DNSResponse]:
        """Query other nodes for DNS record"""
        # Generate query ID
        query_id = self._get_next_query_id()
        
        # Create query
        query = DNSQuery(
            name=name,
            record_type=record_type,
            query_id=query_id
        )
        
        # Create future for response
        future = asyncio.Future()
        self.pending_queries[query_id] = future
        
        try:
            # Send query to neighbors
            await self._send_dns_query(query)
            
            # Wait for response
            response = await asyncio.wait_for(future, timeout)
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"DNS query timeout for {name}")
            return None
        finally:
            self.pending_queries.pop(query_id, None)
    
    def _get_next_query_id(self) -> int:
        """Get next query ID"""
        self.query_id_counter += 1
        return self.query_id_counter
    
    async def _send_dns_query(self, query: DNSQuery):
        """Send DNS query to neighbors"""
        message = {
            'type': 'dns_query',
            'query': {
                'name': query.name,
                'type': query.record_type.value,
                'id': query.query_id,
                'flags': query.flags
            }
        }
        
        # Send to connected peers
        # Could implement anycast or specific resolver selection
        await self.node.broadcast_message(message)
    
    async def handle_dns_query(self, data: Dict):
        """Handle incoming DNS query"""
        try:
            query_data = data['query']
            query = DNSQuery(
                name=query_data['name'],
                record_type=DNSRecordType(query_data['type']),
                query_id=query_data['id'],
                flags=query_data.get('flags', 0)
            )
            
            sender_id = data.get('sender_id')
            
            # Try to resolve
            response = await self.resolve(query.name, query.record_type)
            
            if response:
                # Send response back
                await self._send_dns_response(response, query, sender_id)
            
        except Exception as e:
            logger.error(f"Error handling DNS query: {e}")
    
    async def _send_dns_response(self, response: DNSResponse, 
                                query: DNSQuery, target: NodeID):
        """Send DNS response"""
        message = {
            'type': 'dns_response',
            'query_id': query.query_id,
            'response': {
                'records': [
                    {
                        'name': r.name,
                        'type': r.record_type,
                        'value': r.value,
                        'ttl': r.ttl,
                        'signature': r.signature.hex() if r.signature else None
                    }
                    for r in response.records
                ],
                'authoritative': response.authoritative,
                'rcode': response.rcode
            }
        }
        
        await self.node.send_message(target, message)
    
    async def handle_dns_response(self, data: Dict):
        """Handle incoming DNS response"""
        try:
            query_id = data['query_id']
            
            if query_id in self.pending_queries:
                response_data = data['response']
                
                # Parse records
                records = []
                for r in response_data['records']:
                    record = DNSRecord(
                        name=r['name'],
                        record_type=r['type'],
                        value=r['value'],
                        ttl=r['ttl'],
                        signature=bytes.fromhex(r['signature']) 
                        if r.get('signature') else None
                    )
                    records.append(record)
                
                # Create response
                # Note: We don't have the original query here
                response = DNSResponse(
                    query=None,  # Will be set by resolver
                    records=records,
                    authoritative=response_data.get('authoritative', False),
                    rcode=response_data.get('rcode', 0)
                )
                
                # Complete future
                self.pending_queries[query_id].set_result(response)
            
        except Exception as e:
            logger.error(f"Error handling DNS response: {e}")
    
    def _cache_response(self, response: DNSResponse):
        """Cache DNS response"""
        if not response.query:
            return
        
        cache_key = (response.query.name, response.query.record_type)
        
        # Add to cache
        self.cache[cache_key] = response
        
        # Update cache order
        if cache_key in self.cache_order:
            self.cache_order.remove(cache_key)
        self.cache_order.append(cache_key)
        
        # Enforce cache size limit
        while len(self.cache) > self.config.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
    
    async def update_record(self, name: str, record_type: DNSRecordType,
                          value: str, ttl: int = 3600):
        """Update DNS record (must be authoritative)"""
        if name not in self.authoritative_zones:
            raise ValueError(f"Not authoritative for {name}")
        
        zone = self.authoritative_zones[name]
        
        # Find existing record
        existing = None
        for r in zone.records:
            if r.name == name and r.record_type == record_type.value:
                existing = r
                break
        
        if existing:
            # Update existing
            existing.value = value
            existing.ttl = ttl
            existing.created = datetime.now()
        else:
            # Create new
            record = DNSRecord(
                name=name,
                record_type=record_type.value,
                value=value,
                ttl=ttl
            )
            zone.records.append(record)
        
        # Update serial
        zone.serial += 1
        
        # Re-sign zone
        self._sign_zone(zone)
        
        # Publish updates
        await self._publish_zone(zone)
    
    async def _cache_cleanup_loop(self):
        """Periodically clean expired cache entries"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Find expired entries
                expired = []
                for key, response in self.cache.items():
                    if any(r.is_expired() for r in response.records):
                        expired.append(key)
                
                # Remove expired
                for key in expired:
                    del self.cache[key]
                    self.cache_order.remove(key)
                
                if expired:
                    logger.debug(f"Cleaned {len(expired)} expired DNS cache entries")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    async def _zone_refresh_loop(self):
        """Periodically refresh zones from DHT"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Republish our zones
                for zone in self.authoritative_zones.values():
                    await self._publish_zone(zone)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in zone refresh: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DNS overlay statistics"""
        return {
            'authoritative_zones': len(self.authoritative_zones),
            'cached_records': len(self.cache),
            'pending_queries': len(self.pending_queries),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'root_domain': self.ROOT_DOMAIN
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # Would track hits/misses in production
        return 0.0