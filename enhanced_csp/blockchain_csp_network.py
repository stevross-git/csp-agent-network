#!/usr/bin/env python3
"""
Blockchain CSP Network
======================

Decentralized AI-to-AI communication network using blockchain technology:
- Distributed CSP process registry
- Smart contracts for AI agreements
- Consensus mechanisms for process validation
- Tokenized AI services and capabilities
- Immutable audit trails for AI interactions
- Decentralized identity for AI agents
- Cross-chain interoperability
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import logging
from collections import defaultdict, deque
import uuid
import ecdsa
import base58
from datetime import datetime, timezone
import requests
import websockets

# Blockchain libraries
try:
    from web3 import Web3
    from eth_account import Account
    from eth_account.messages import encode_defunct
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logging.warning("Web3 not available - using blockchain simulation")

# Import our CSP components
from core.advanced_csp_core import Process, ProcessContext, Channel, Event
from ai_integration.csp_ai_integration import AIAgent

# ============================================================================
# BLOCKCHAIN PRIMITIVES
# ============================================================================

@dataclass
class Block:
    """Blockchain block containing CSP transactions"""
    index: int
    timestamp: float
    transactions: List[Dict[str, Any]]
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    validator: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': self.transactions,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 4):
        """Mine block using proof of work"""
        target = "0" * difficulty
        
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()
    
    def is_valid(self) -> bool:
        """Validate block integrity"""
        return self.hash == self.calculate_hash()

@dataclass
class Transaction:
    """Blockchain transaction for CSP operations"""
    transaction_id: str
    sender: str
    receiver: str
    transaction_type: str  # 'process_registry', 'ai_agreement', 'capability_transfer'
    payload: Dict[str, Any]
    timestamp: float
    signature: str = ""
    gas_fee: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return asdict(self)
    
    def sign_transaction(self, private_key: str):
        """Sign transaction with private key"""
        message = json.dumps({
            'transaction_id': self.transaction_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'transaction_type': self.transaction_type,
            'payload': self.payload,
            'timestamp': self.timestamp
        }, sort_keys=True)
        
        # Simple signature simulation
        signature_hash = hashlib.sha256(f"{message}{private_key}".encode()).hexdigest()
        self.signature = signature_hash
    
    def verify_signature(self, public_key: str) -> bool:
        """Verify transaction signature"""
        message = json.dumps({
            'transaction_id': self.transaction_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'transaction_type': self.transaction_type,
            'payload': self.payload,
            'timestamp': self.timestamp
        }, sort_keys=True)
        
        expected_signature = hashlib.sha256(f"{message}{public_key}".encode()).hexdigest()
        return self.signature == expected_signature

class ConsensusAlgorithm(Enum):
    """Different consensus algorithms"""
    PROOF_OF_WORK = auto()
    PROOF_OF_STAKE = auto()
    PRACTICAL_BYZANTINE_FAULT_TOLERANCE = auto()
    DELEGATED_PROOF_OF_STAKE = auto()

# ============================================================================
# BLOCKCHAIN CSP NETWORK
# ============================================================================

class BlockchainCSPNetwork:
    """Decentralized CSP network using blockchain"""
    
    def __init__(self, node_id: str, consensus: ConsensusAlgorithm = ConsensusAlgorithm.PROOF_OF_STAKE):
        self.node_id = node_id
        self.consensus_algorithm = consensus
        self.blockchain = [self._create_genesis_block()]
        self.pending_transactions = []
        self.peers = {}
        self.ai_agents = {}
        self.smart_contracts = {}
        self.stake_pool = {}
        self.reputation_scores = defaultdict(float)
        self.gas_price = 0.001  # Base gas price
        
        # Crypto wallet simulation
        self.private_key = self._generate_private_key()
        self.public_key = self._generate_public_key()
        self.wallet_address = self._generate_address()
        
        # Network state
        self.network_state = {
            'total_processes': 0,
            'active_agreements': 0,
            'total_value_locked': 0.0,
            'network_health': 1.0
        }
    
    def _create_genesis_block(self) -> Block:
        """Create the genesis block"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[{
                'type': 'genesis',
                'message': 'CSP Network Genesis Block',
                'network_id': 'csp-mainnet-v1'
            }],
            previous_hash="0"
        )
        genesis_block.hash = genesis_block.calculate_hash()
        return genesis_block
    
    def _generate_private_key(self) -> str:
        """Generate private key for node"""
        return hashlib.sha256(f"{self.node_id}{time.time()}".encode()).hexdigest()
    
    def _generate_public_key(self) -> str:
        """Generate public key from private key"""
        return hashlib.sha256(f"public_{self.private_key}".encode()).hexdigest()
    
    def _generate_address(self) -> str:
        """Generate wallet address"""
        return hashlib.sha256(f"addr_{self.public_key}".encode()).hexdigest()[:40]
    
    async def register_ai_agent(self, agent: AIAgent, capabilities: List[str], 
                               stake_amount: float = 1.0) -> str:
        """Register AI agent on blockchain"""
        
        # Create registration transaction
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            sender=self.wallet_address,
            receiver="network_registry",
            transaction_type="ai_agent_registration",
            payload={
                'agent_id': agent.name,
                'capabilities': capabilities,
                'stake_amount': stake_amount,
                'reputation_score': 0.0,
                'registration_timestamp': time.time()
            },
            timestamp=time.time(),
            gas_fee=self.gas_price * 100
        )
        
        transaction.sign_transaction(self.private_key)
        await self._add_transaction(transaction)
        
        # Update local registry
        self.ai_agents[agent.name] = {
            'agent': agent,
            'capabilities': capabilities,
            'stake_amount': stake_amount,
            'reputation': 0.0,
            'active_agreements': [],
            'total_interactions': 0
        }
        
        return transaction.transaction_id
    
    async def create_ai_agreement(self, agent1_id: str, agent2_id: str, 
                                 agreement_terms: Dict[str, Any]) -> str:
        """Create smart contract agreement between AI agents"""
        
        agreement_id = str(uuid.uuid4())
        
        # Create smart contract
        smart_contract = AISmartContract(
            contract_id=agreement_id,
            parties=[agent1_id, agent2_id],
            terms=agreement_terms,
            creator=self.wallet_address
        )
        
        # Deploy contract transaction
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            sender=self.wallet_address,
            receiver="smart_contract_registry",
            transaction_type="deploy_ai_agreement",
            payload={
                'agreement_id': agreement_id,
                'parties': [agent1_id, agent2_id],
                'terms': agreement_terms,
                'contract_code': smart_contract.get_bytecode(),
                'creation_timestamp': time.time()
            },
            timestamp=time.time(),
            gas_fee=self.gas_price * 500
        )
        
        transaction.sign_transaction(self.private_key)
        await self._add_transaction(transaction)
        
        # Store contract locally
        self.smart_contracts[agreement_id] = smart_contract
        
        # Update agent records
        if agent1_id in self.ai_agents:
            self.ai_agents[agent1_id]['active_agreements'].append(agreement_id)
        if agent2_id in self.ai_agents:
            self.ai_agents[agent2_id]['active_agreements'].append(agreement_id)
        
        return agreement_id
    
    async def execute_ai_interaction(self, agreement_id: str, 
                                   interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI interaction through smart contract"""
        
        if agreement_id not in self.smart_contracts:
            raise ValueError("Agreement not found")
        
        contract = self.smart_contracts[agreement_id]
        
        # Execute contract logic
        execution_result = await contract.execute_interaction(interaction_data)
        
        # Record interaction on blockchain
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            sender=self.wallet_address,
            receiver=agreement_id,
            transaction_type="ai_interaction",
            payload={
                'agreement_id': agreement_id,
                'interaction_data': interaction_data,
                'execution_result': execution_result,
                'gas_used': execution_result.get('gas_used', 50),
                'timestamp': time.time()
            },
            timestamp=time.time(),
            gas_fee=self.gas_price * execution_result.get('gas_used', 50)
        )
        
        transaction.sign_transaction(self.private_key)
        await self._add_transaction(transaction)
        
        # Update reputation scores
        await self._update_reputation_scores(agreement_id, execution_result)
        
        return execution_result
    
    async def _add_transaction(self, transaction: Transaction):
        """Add transaction to pending pool"""
        self.pending_transactions.append(transaction)
        
        # Auto-mine blocks when enough transactions
        if len(self.pending_transactions) >= 10:
            await self._mine_block()
    
    async def _mine_block(self):
        """Mine a new block"""
        if not self.pending_transactions:
            return
        
        # Create new block
        new_block = Block(
            index=len(self.blockchain),
            timestamp=time.time(),
            transactions=[tx.to_dict() for tx in self.pending_transactions],
            previous_hash=self.blockchain[-1].hash,
            validator=self.wallet_address
        )
        
        # Apply consensus algorithm
        if self.consensus_algorithm == ConsensusAlgorithm.PROOF_OF_WORK:
            new_block.mine_block(difficulty=4)
        elif self.consensus_algorithm == ConsensusAlgorithm.PROOF_OF_STAKE:
            # Simplified PoS - just set hash
            new_block.hash = new_block.calculate_hash()
        
        # Add block to chain
        self.blockchain.append(new_block)
        
        # Clear pending transactions
        self.pending_transactions = []
        
        # Update network state
        await self._update_network_state()
        
        logging.info(f"Mined block #{new_block.index} with {len(new_block.transactions)} transactions")
    
    async def _update_network_state(self):
        """Update network state metrics"""
        self.network_state['total_processes'] = len(self.ai_agents)
        self.network_state['active_agreements'] = len(self.smart_contracts)
        self.network_state['total_value_locked'] = sum(
            agent_data['stake_amount'] for agent_data in self.ai_agents.values()
        )
        
        # Calculate network health based on various factors
        health_factors = []
        
        # Factor 1: Active participation
        if self.ai_agents:
            avg_interactions = sum(
                agent_data['total_interactions'] for agent_data in self.ai_agents.values()
            ) / len(self.ai_agents)
            health_factors.append(min(avg_interactions / 100, 1.0))
        
        # Factor 2: Reputation scores
        if self.reputation_scores:
            avg_reputation = sum(self.reputation_scores.values()) / len(self.reputation_scores)
            health_factors.append(avg_reputation)
        
        # Factor 3: Smart contract success rate
        health_factors.append(0.95)  # Placeholder
        
        self.network_state['network_health'] = sum(health_factors) / len(health_factors) if health_factors else 1.0
    
    async def _update_reputation_scores(self, agreement_id: str, execution_result: Dict[str, Any]):
        """Update reputation scores based on interaction results"""
        if agreement_id not in self.smart_contracts:
            return
        
        contract = self.smart_contracts[agreement_id]
        success_rate = execution_result.get('success_rate', 0.5)
        
        # Update reputation for all parties
        for party in contract.parties:
            if party in self.ai_agents:
                current_reputation = self.reputation_scores[party]
                # Exponential moving average
                self.reputation_scores[party] = 0.9 * current_reputation + 0.1 * success_rate
                self.ai_agents[party]['reputation'] = self.reputation_scores[party]
                self.ai_agents[party]['total_interactions'] += 1
    
    async def validate_blockchain(self) -> bool:
        """Validate entire blockchain integrity"""
        for i in range(1, len(self.blockchain)):
            current_block = self.blockchain[i]
            previous_block = self.blockchain[i-1]
            
            # Check hash integrity
            if not current_block.is_valid():
                return False
            
            # Check chain linkage
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    async def sync_with_peer(self, peer_address: str):
        """Synchronize blockchain with a peer node"""
        try:
            # Simulate peer communication
            logging.info(f"Syncing with peer: {peer_address}")
            
            # In a real implementation, this would:
            # 1. Request peer's blockchain
            # 2. Validate incoming blocks
            # 3. Resolve conflicts using longest chain rule
            # 4. Update local state
            
            peer_id = peer_address.split(':')[0]
            self.peers[peer_id] = {
                'address': peer_address,
                'last_sync': time.time(),
                'status': 'active'
            }
            
            return True
            
        except Exception as e:
            logging.error(f"Peer sync failed: {e}")
            return False
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get comprehensive blockchain statistics"""
        total_transactions = sum(len(block.transactions) for block in self.blockchain)
        
        # Analyze transaction types
        transaction_types = defaultdict(int)
        for block in self.blockchain:
            for tx in block.transactions:
                tx_type = tx.get('transaction_type', 'unknown')
                transaction_types[tx_type] += 1
        
        # Calculate network metrics
        stats = {
            'blockchain_height': len(self.blockchain),
            'total_transactions': total_transactions,
            'transaction_types': dict(transaction_types),
            'pending_transactions': len(self.pending_transactions),
            'registered_agents': len(self.ai_agents),
            'active_contracts': len(self.smart_contracts),
            'network_peers': len(self.peers),
            'network_state': self.network_state,
            'consensus_algorithm': self.consensus_algorithm.name,
            'avg_block_time': self._calculate_avg_block_time(),
            'chain_integrity': asyncio.create_task(self.validate_blockchain())
        }
        
        return stats
    
    def _calculate_avg_block_time(self) -> float:
        """Calculate average block mining time"""
        if len(self.blockchain) < 2:
            return 0.0
        
        total_time = 0
        for i in range(1, len(self.blockchain)):
            block_time = self.blockchain[i].timestamp - self.blockchain[i-1].timestamp
            total_time += block_time
        
        return total_time / (len(self.blockchain) - 1)

# ============================================================================
# AI SMART CONTRACTS
# ============================================================================

class AISmartContract:
    """Smart contract for AI agent agreements"""
    
    def __init__(self, contract_id: str, parties: List[str], 
                 terms: Dict[str, Any], creator: str):
        self.contract_id = contract_id
        self.parties = parties
        self.terms = terms
        self.creator = creator
        self.creation_time = time.time()
        self.state = 'active'
        self.execution_history = []
        self.gas_limit = terms.get('gas_limit', 1000000)
        self.value_locked = terms.get('value_locked', 0.0)
    
    def get_bytecode(self) -> str:
        """Get contract bytecode (simplified)"""
        contract_data = {
            'contract_id': self.contract_id,
            'parties': self.parties,
            'terms': self.terms,
            'creator': self.creator,
            'creation_time': self.creation_time
        }
        return hashlib.sha256(json.dumps(contract_data, sort_keys=True).encode()).hexdigest()
    
    async def execute_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI interaction according to contract terms"""
        
        execution_start = time.time()
        gas_used = 0
        
        try:
            # Validate interaction
            if not self._validate_interaction(interaction_data):
                return {
                    'success': False,
                    'error': 'Invalid interaction data',
                    'gas_used': 21000  # Base gas for failed transaction
                }
            
            # Execute contract logic
            result = await self._execute_contract_logic(interaction_data)
            gas_used = result.get('gas_used', 50000)
            
            # Update contract state
            self.execution_history.append({
                'timestamp': execution_start,
                'interaction_data': interaction_data,
                'result': result,
                'gas_used': gas_used
            })
            
            return {
                'success': True,
                'result': result,
                'gas_used': gas_used,
                'execution_time': time.time() - execution_start,
                'success_rate': result.get('success_rate', 1.0)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'gas_used': gas_used + 21000,
                'execution_time': time.time() - execution_start,
                'success_rate': 0.0
            }
    
    def _validate_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """Validate interaction data against contract terms"""
        
        # Check required fields
        required_fields = self.terms.get('required_fields', [])
        for field in required_fields:
            if field not in interaction_data:
                return False
        
        # Check data constraints
        constraints = self.terms.get('constraints', {})
        for field, constraint in constraints.items():
            if field in interaction_data:
                value = interaction_data[field]
                if not self._check_constraint(value, constraint):
                    return False
        
        return True
    
    def _check_constraint(self, value: Any, constraint: Dict[str, Any]) -> bool:
        """Check if value satisfies constraint"""
        
        if 'type' in constraint:
            expected_type = constraint['type']
            if expected_type == 'string' and not isinstance(value, str):
                return False
            elif expected_type == 'number' and not isinstance(value, (int, float)):
                return False
            elif expected_type == 'boolean' and not isinstance(value, bool):
                return False
        
        if 'min_value' in constraint and value < constraint['min_value']:
            return False
        
        if 'max_value' in constraint and value > constraint['max_value']:
            return False
        
        if 'allowed_values' in constraint and value not in constraint['allowed_values']:
            return False
        
        return True
    
    async def _execute_contract_logic(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main contract logic"""
        
        contract_type = self.terms.get('contract_type', 'collaboration')
        
        if contract_type == 'collaboration':
            return await self._execute_collaboration_contract(interaction_data)
        elif contract_type == 'competition':
            return await self._execute_competition_contract(interaction_data)
        elif contract_type == 'data_exchange':
            return await self._execute_data_exchange_contract(interaction_data)
        elif contract_type == 'service_agreement':
            return await self._execute_service_agreement_contract(interaction_data)
        else:
            return await self._execute_generic_contract(interaction_data)
    
    async def _execute_collaboration_contract(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute collaboration contract"""
        
        # Simulate collaborative AI task
        task_complexity = interaction_data.get('task_complexity', 0.5)
        collaboration_quality = interaction_data.get('collaboration_quality', 0.8)
        
        # Calculate success rate
        success_rate = min(collaboration_quality * (1 - task_complexity * 0.3), 1.0)
        
        # Calculate rewards
        base_reward = self.terms.get('base_reward', 10.0)
        reward_per_party = base_reward * success_rate / len(self.parties)
        
        return {
            'contract_type': 'collaboration',
            'success_rate': success_rate,
            'reward_per_party': reward_per_party,
            'total_reward': base_reward * success_rate,
            'gas_used': int(50000 * (1 + task_complexity)),
            'collaboration_score': collaboration_quality
        }
    
    async def _execute_competition_contract(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute competition contract"""
        
        # Simulate competitive AI task
        performance_scores = interaction_data.get('performance_scores', {})
        
        if not performance_scores:
            return {'error': 'No performance scores provided', 'gas_used': 30000}
        
        # Determine winner
        winner = max(performance_scores, key=performance_scores.get)
        winner_score = performance_scores[winner]
        
        # Calculate rewards
        total_prize = self.terms.get('total_prize', 100.0)
        winner_reward = total_prize * 0.7  # Winner gets 70%
        participation_reward = total_prize * 0.3 / (len(self.parties) - 1)
        
        return {
            'contract_type': 'competition',
            'winner': winner,
            'winner_score': winner_score,
            'winner_reward': winner_reward,
            'participation_reward': participation_reward,
            'performance_scores': performance_scores,
            'gas_used': 75000,
            'success_rate': winner_score
        }
    
    async def _execute_data_exchange_contract(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data exchange contract"""
        
        # Simulate data exchange
        data_quality = interaction_data.get('data_quality', 0.8)
        data_size = interaction_data.get('data_size', 1000)  # bytes
        
        # Calculate transaction fee
        fee_per_kb = self.terms.get('fee_per_kb', 0.001)
        total_fee = (data_size / 1024) * fee_per_kb
        
        return {
            'contract_type': 'data_exchange',
            'data_quality': data_quality,
            'data_size': data_size,
            'total_fee': total_fee,
            'success_rate': data_quality,
            'gas_used': int(30000 + data_size / 10),
            'exchange_timestamp': time.time()
        }
    
    async def _execute_service_agreement_contract(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute service agreement contract"""
        
        # Simulate service execution
        service_quality = interaction_data.get('service_quality', 0.9)
        service_duration = interaction_data.get('service_duration', 1.0)  # hours
        
        # Calculate service fee
        hourly_rate = self.terms.get('hourly_rate', 5.0)
        total_fee = service_duration * hourly_rate * service_quality
        
        return {
            'contract_type': 'service_agreement',
            'service_quality': service_quality,
            'service_duration': service_duration,
            'total_fee': total_fee,
            'success_rate': service_quality,
            'gas_used': int(40000 + service_duration * 1000),
            'service_timestamp': time.time()
        }
    
    async def _execute_generic_contract(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic contract"""
        
        # Basic execution
        success_rate = interaction_data.get('success_rate', 0.8)
        
        return {
            'contract_type': 'generic',
            'success_rate': success_rate,
            'gas_used': 35000,
            'execution_timestamp': time.time()
        }

# ============================================================================
# CROSS-CHAIN INTEROPERABILITY
# ============================================================================

class CrossChainBridge:
    """Bridge for cross-chain CSP operations"""
    
    def __init__(self, supported_chains: List[str]):
        self.supported_chains = supported_chains
        self.bridge_contracts = {}
        self.pending_transfers = {}
        self.transfer_history = []
    
    async def transfer_ai_agent(self, agent_id: str, source_chain: str, 
                               target_chain: str) -> str:
        """Transfer AI agent between chains"""
        
        if source_chain not in self.supported_chains or target_chain not in self.supported_chains:
            raise ValueError("Unsupported chain")
        
        transfer_id = str(uuid.uuid4())
        
        # Lock agent on source chain
        lock_result = await self._lock_agent_on_source(agent_id, source_chain)
        
        if lock_result['success']:
            # Mint agent on target chain
            mint_result = await self._mint_agent_on_target(agent_id, target_chain, lock_result['agent_data'])
            
            if mint_result['success']:
                # Complete transfer
                transfer_record = {
                    'transfer_id': transfer_id,
                    'agent_id': agent_id,
                    'source_chain': source_chain,
                    'target_chain': target_chain,
                    'timestamp': time.time(),
                    'status': 'completed'
                }
                
                self.transfer_history.append(transfer_record)
                return transfer_id
        
        # If we get here, transfer failed
        raise Exception("Cross-chain transfer failed")
    
    async def _lock_agent_on_source(self, agent_id: str, chain: str) -> Dict[str, Any]:
        """Lock agent on source chain"""
        # Simulate locking mechanism
        return {
            'success': True,
            'agent_data': {
                'agent_id': agent_id,
                'capabilities': ['reasoning', 'learning'],
                'reputation': 0.85,
                'stake_amount': 10.0
            }
        }
    
    async def _mint_agent_on_target(self, agent_id: str, chain: str, 
                                   agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mint agent on target chain"""
        # Simulate minting mechanism
        return {
            'success': True,
            'new_agent_address': f"{chain}:{agent_id}",
            'minted_data': agent_data
        }

# ============================================================================
# DECENTRALIZED IDENTITY FOR AI AGENTS
# ============================================================================

class DecentralizedIdentity:
    """Decentralized identity system for AI agents"""
    
    def __init__(self):
        self.identity_registry = {}
        self.credential_schemas = {}
        self.verified_credentials = {}
    
    async def create_identity(self, agent_id: str, identity_data: Dict[str, Any]) -> str:
        """Create decentralized identity for AI agent"""
        
        # Generate DID (Decentralized Identifier)
        did = f"did:csp:{hashlib.sha256(agent_id.encode()).hexdigest()[:16]}"
        
        # Create identity document
        identity_document = {
            'id': did,
            'agent_id': agent_id,
            'created': datetime.now(timezone.utc).isoformat(),
            'updated': datetime.now(timezone.utc).isoformat(),
            'publicKey': [{
                'id': f"{did}#key1",
                'type': 'EcdsaSecp256k1VerificationKey2019',
                'controller': did,
                'publicKeyHex': hashlib.sha256(f"pubkey_{agent_id}".encode()).hexdigest()
            }],
            'authentication': [f"{did}#key1"],
            'service': [{
                'id': f"{did}#agent-service",
                'type': 'CSPAgentService',
                'serviceEndpoint': f"https://csp-network.com/agents/{agent_id}"
            }],
            'capabilities': identity_data.get('capabilities', []),
            'certifications': identity_data.get('certifications', [])
        }
        
        # Store identity
        self.identity_registry[did] = identity_document
        
        return did
    
    async def issue_credential(self, issuer_did: str, subject_did: str, 
                              credential_type: str, claims: Dict[str, Any]) -> str:
        """Issue verifiable credential"""
        
        credential_id = str(uuid.uuid4())
        
        credential = {
            'id': credential_id,
            'type': ['VerifiableCredential', credential_type],
            'issuer': issuer_did,
            'subject': subject_did,
            'issuanceDate': datetime.now(timezone.utc).isoformat(),
            'claims': claims,
            'signature': hashlib.sha256(f"{issuer_did}{subject_did}{json.dumps(claims)}".encode()).hexdigest()
        }
        
        self.verified_credentials[credential_id] = credential
        
        return credential_id
    
    async def verify_credential(self, credential_id: str) -> Dict[str, Any]:
        """Verify a credential"""
        
        if credential_id not in self.verified_credentials:
            return {'valid': False, 'error': 'Credential not found'}
        
        credential = self.verified_credentials[credential_id]
        
        # Verify signature (simplified)
        expected_signature = hashlib.sha256(
            f"{credential['issuer']}{credential['subject']}{json.dumps(credential['claims'])}".encode()
        ).hexdigest()
        
        valid = credential['signature'] == expected_signature
        
        return {
            'valid': valid,
            'credential': credential if valid else None,
            'verification_timestamp': time.time()
        }

# ============================================================================
# BLOCKCHAIN CSP DEMO
# ============================================================================

async def blockchain_csp_demo():
    """Demonstrate blockchain CSP network capabilities"""
    
    print("🔗 Blockchain CSP Network Demo")
    print("=" * 50)
    
    # Create blockchain network
    network = BlockchainCSPNetwork("node_001", ConsensusAlgorithm.PROOF_OF_STAKE)
    
    print(f"✅ Created blockchain CSP network (Node: {network.node_id})")
    print(f"   Wallet Address: {network.wallet_address}")
    
    # Create AI agents
    from ai_integration.csp_ai_integration import AIAgent, LLMCapability
    
    alice_capability = LLMCapability("gpt-4", "reasoning")
    bob_capability = LLMCapability("claude-3", "analysis")
    
    alice_agent = AIAgent("Alice_AI", [alice_capability])
    bob_agent = AIAgent("Bob_AI", [bob_capability])
    
    # Register AI agents on blockchain
    alice_tx = await network.register_ai_agent(alice_agent, ["reasoning", "problem_solving"], 10.0)
    bob_tx = await network.register_ai_agent(bob_agent, ["analysis", "data_processing"], 15.0)
    
    print(f"✅ Registered Alice_AI (TX: {alice_tx[:8]}...)")
    print(f"✅ Registered Bob_AI (TX: {bob_tx[:8]}...)")
    
    # Create AI agreement
    agreement_terms = {
        'contract_type': 'collaboration',
        'base_reward': 20.0,
        'duration': 3600,  # 1 hour
        'required_fields': ['task_complexity', 'collaboration_quality'],
        'gas_limit': 100000
    }
    
    agreement_id = await network.create_ai_agreement("Alice_AI", "Bob_AI", agreement_terms)
    print(f"✅ Created AI agreement (ID: {agreement_id[:8]}...)")
    
    # Execute AI interactions
    for i in range(3):
        interaction_data = {
            'task_complexity': 0.3 + i * 0.2,
            'collaboration_quality': 0.8 + i * 0.05,
            'task_id': f"task_{i+1}"
        }
        
        result = await network.execute_ai_interaction(agreement_id, interaction_data)
        print(f"✅ Executed interaction {i+1}: Success rate {result['success_rate']:.2f}, "
              f"Reward: {result['result']['reward_per_party']:.2f}")
    
    # Mine remaining transactions
    await network._mine_block()
    
    # Get blockchain statistics
    stats = network.get_blockchain_stats()
    print(f"\n📊 Blockchain Statistics:")
    print(f"   Blockchain Height: {stats['blockchain_height']}")
    print(f"   Total Transactions: {stats['total_transactions']}")
    print(f"   Registered Agents: {stats['registered_agents']}")
    print(f"   Active Contracts: {stats['active_contracts']}")
    print(f"   Network Health: {stats['network_state']['network_health']:.2f}")
    
    # Demonstrate cross-chain transfer
    bridge = CrossChainBridge(['ethereum', 'polygon', 'bsc'])
    transfer_id = await bridge.transfer_ai_agent("Alice_AI", "ethereum", "polygon")
    print(f"✅ Cross-chain transfer: {transfer_id[:8]}...")
    
    # Demonstrate decentralized identity
    identity_system = DecentralizedIdentity()
    alice_did = await identity_system.create_identity("Alice_AI", {
        'capabilities': ['reasoning', 'problem_solving'],
        'certifications': ['AI_Safety_Level_3']
    })
    
    credential_id = await identity_system.issue_credential(
        "did:csp:network_authority", alice_did, "AICapabilityCredential",
        {'reasoning_score': 0.95, 'safety_rating': 'AAA'}
    )
    
    verification = await identity_system.verify_credential(credential_id)
    print(f"✅ Created DID: {alice_did}")
    print(f"✅ Issued credential: {credential_id[:8]}... (Valid: {verification['valid']})")
    
    # Validate blockchain
    is_valid = await network.validate_blockchain()
    print(f"✅ Blockchain validation: {'PASSED' if is_valid else 'FAILED'}")
    
    print("\n🎉 Blockchain CSP Demo completed successfully!")
    print("Features demonstrated:")
    print("• Decentralized AI agent registration")
    print("• Smart contract agreements between AI agents")
    print("• Automated execution and reward distribution")
    print("• Reputation scoring and tracking")
    print("• Blockchain integrity and validation")
    print("• Cross-chain interoperability")
    print("• Decentralized identity for AI agents")
    print("• Verifiable credentials and attestations")
    print("• Consensus mechanisms (Proof of Stake)")
    print("• Comprehensive network monitoring")

if __name__ == "__main__":
    asyncio.run(blockchain_csp_demo())
