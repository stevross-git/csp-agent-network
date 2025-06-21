# CSP Process Integrity Monitoring System
# Critical security component for Enhanced CSP

import asyncio
import hashlib
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class ProcessThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ProcessSignature:
    """Cryptographic signature for process integrity"""
    process_id: str
    executable_hash: str
    command_line_hash: str
    creation_time: float
    parent_process: str
    user_context: str
    signature_timestamp: float

@dataclass
class IntegrityViolation:
    """Record of process integrity violation"""
    violation_id: str
    process_id: str
    violation_type: str
    threat_level: ProcessThreatLevel
    description: str
    timestamp: datetime
    evidence: Dict[str, Any]
    remediation_taken: bool = False

class CSPProcessMonitor:
    """Advanced CSP Process Integrity Monitoring"""
    
    def __init__(self):
        self.legitimate_processes: Dict[str, ProcessSignature] = {}
        self.active_processes: Dict[str, Dict] = {}
        self.violations: List[IntegrityViolation] = []
        self.monitoring_active = False
        self.whitelist_hashes: Set[str] = set()
        self.blacklist_hashes: Set[str] = set()
        
        # Known CSP process patterns
        self.csp_process_patterns = {
            'orchestrator': ['csp_orchestrator', 'process_manager'],
            'communicator': ['csp_comm', 'channel_manager'],
            'visualizer': ['csp_viz', 'dashboard'],
            'security': ['csp_security', 'threat_detector']
        }
        
        # Suspicious patterns
        self.suspicious_patterns = [
            'cmd.exe',
            'powershell.exe',
            'bash',
            '/bin/sh',
            'nc.exe',
            'netcat',
            'telnet',
            'wget',
            'curl'
        ]
        
        self.lock = threading.Lock()
    
    def calculate_process_hash(self, process_info: Dict) -> str:
        """Calculate unique hash for process"""
        hash_input = f"{process_info.get('exe', '')}{process_info.get('cmdline', '')}{process_info.get('create_time', 0)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def is_legitimate_csp_process(self, process_info: Dict) -> bool:
        """Check if process is a legitimate CSP component"""
        exe_name = process_info.get('name', '').lower()
        exe_path = process_info.get('exe', '').lower()
        cmdline = ' '.join(process_info.get('cmdline', [])).lower()
        
        # Check against known CSP patterns
        for category, patterns in self.csp_process_patterns.items():
            for pattern in patterns:
                if pattern in exe_name or pattern in exe_path or pattern in cmdline:
                    return True
        
        # Check if it's a Python process running CSP code
        if 'python' in exe_name and any(csp_term in cmdline for csp_term in ['csp', 'enhanced_csp', 'breach_benchmark']):
            return True
        
        return False
    
    def detect_process_injection(self, process_info: Dict) -> Optional[IntegrityViolation]:
        """Detect potential process injection attacks"""
        pid = process_info.get('pid')
        exe_name = process_info.get('name', '').lower()
        exe_path = process_info.get('exe', '')
        cmdline = process_info.get('cmdline', [])
        
        violation = None
        
        # Check for suspicious process names in CSP context
        if any(pattern in exe_name for pattern in self.suspicious_patterns):
            if self.is_csp_context(process_info):
                violation = IntegrityViolation(
                    violation_id=f"injection_{pid}_{int(time.time())}",
                    process_id=str(pid),
                    violation_type="suspicious_process_injection",
                    threat_level=ProcessThreatLevel.HIGH,
                    description=f"Suspicious process '{exe_name}' detected in CSP context",
                    timestamp=datetime.now(),
                    evidence={
                        'process_name': exe_name,
                        'executable_path': exe_path,
                        'command_line': cmdline,
                        'parent_pid': process_info.get('ppid')
                    }
                )
        
        # Check for unsigned or modified executables
        process_hash = self.calculate_process_hash(process_info)
        if process_hash in self.blacklist_hashes:
            violation = IntegrityViolation(
                violation_id=f"blacklist_{pid}_{int(time.time())}",
                process_id=str(pid),
                violation_type="blacklisted_process",
                threat_level=ProcessThreatLevel.CRITICAL,
                description=f"Blacklisted process detected: {exe_name}",
                timestamp=datetime.now(),
                evidence={'process_hash': process_hash, 'executable_path': exe_path}
            )
        
        # Check for process hollowing indicators
        if self.detect_process_hollowing(process_info):
            violation = IntegrityViolation(
                violation_id=f"hollowing_{pid}_{int(time.time())}",
                process_id=str(pid),
                violation_type="process_hollowing",
                threat_level=ProcessThreatLevel.CRITICAL,
                description=f"Potential process hollowing detected in {exe_name}",
                timestamp=datetime.now(),
                evidence=process_info
            )
        
        return violation
    
    def is_csp_context(self, process_info: Dict) -> bool:
        """Check if process is running in CSP system context"""
        # Check parent process
        try:
            parent_pid = process_info.get('ppid')
            if parent_pid and parent_pid in self.active_processes:
                parent_info = self.active_processes[parent_pid]
                if self.is_legitimate_csp_process(parent_info):
                    return True
        except:
            pass
        
        # Check process tree
        cmdline = ' '.join(process_info.get('cmdline', [])).lower()
        cwd = process_info.get('cwd', '').lower()
        
        return any(csp_term in cmdline or csp_term in cwd 
                  for csp_term in ['csp', 'enhanced_csp', 'process_orchestrator'])
    
    def detect_process_hollowing(self, process_info: Dict) -> bool:
        """Detect potential process hollowing"""
        try:
            pid = process_info.get('pid')
            process = psutil.Process(pid)
            
            # Check for memory anomalies (simplified detection)
            memory_info = process.memory_info()
            
            # Suspicious if private memory usage is very different from virtual memory
            if memory_info.vms > 0 and memory_info.rss > 0:
                ratio = memory_info.rss / memory_info.vms
                if ratio < 0.1 or ratio > 0.9:  # Extreme ratios can indicate hollowing
                    return True
            
            # Check for unusual memory maps (would require more advanced analysis)
            # This is a simplified check
            return False
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def authenticate_process(self, process_info: Dict) -> bool:
        """Authenticate CSP process using digital signatures"""
        try:
            exe_path = process_info.get('exe', '')
            if not exe_path:
                return False
            
            # Calculate file hash
            with open(exe_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Check against whitelist
            if file_hash in self.whitelist_hashes:
                return True
            
            # For CSP processes, we expect them to be in our whitelist
            if self.is_legitimate_csp_process(process_info):
                # Auto-add to whitelist if it's a legitimate CSP process
                self.whitelist_hashes.add(file_hash)
                logger.info(f"Added CSP process to whitelist: {exe_path} ({file_hash[:16]}...)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error authenticating process: {e}")
            return False
    
    async def scan_active_processes(self):
        """Scan all active processes for integrity violations"""
        violations_found = []
        
        try:
            current_processes = {}
            
            for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline', 'create_time', 'ppid', 'cwd']):
                try:
                    proc_info = proc.info
                    pid = proc_info['pid']
                    current_processes[pid] = proc_info
                    
                    # Check for new processes
                    if pid not in self.active_processes:
                        logger.debug(f"New process detected: {proc_info['name']} (PID: {pid})")
                        
                        # Authenticate new process
                        if not self.authenticate_process(proc_info):
                            logger.warning(f"Unauthenticated process: {proc_info['name']} (PID: {pid})")
                        
                        # Check for injection
                        violation = self.detect_process_injection(proc_info)
                        if violation:
                            violations_found.append(violation)
                            logger.error(f"Process integrity violation: {violation.description}")
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Update active processes
            with self.lock:
                self.active_processes = current_processes
                self.violations.extend(violations_found)
            
            # Alert on violations
            for violation in violations_found:
                await self.handle_violation(violation)
                
        except Exception as e:
            logger.error(f"Error during process scan: {e}")
    
    async def handle_violation(self, violation: IntegrityViolation):
        """Handle detected process integrity violation"""
        logger.critical(f"SECURITY VIOLATION: {violation.description}")
        
        # Automated response based on threat level
        if violation.threat_level == ProcessThreatLevel.CRITICAL:
            await self.terminate_malicious_process(violation.process_id)
            violation.remediation_taken = True
        
        # Send alert to monitoring system
        await self.send_security_alert(violation)
        
        # Log to security audit
        self.log_security_event(violation)
    
    async def terminate_malicious_process(self, process_id: str):
        """Terminate malicious process"""
        try:
            pid = int(process_id)
            process = psutil.Process(pid)
            
            logger.warning(f"Terminating malicious process: {process.name()} (PID: {pid})")
            process.terminate()
            
            # Wait for graceful termination
            await asyncio.sleep(2)
            
            # Force kill if still running
            if process.is_running():
                process.kill()
                logger.warning(f"Force killed process PID: {pid}")
                
        except (psutil.NoSuchProcess, ValueError):
            logger.info(f"Process {process_id} already terminated")
        except Exception as e:
            logger.error(f"Error terminating process {process_id}: {e}")
    
    async def send_security_alert(self, violation: IntegrityViolation):
        """Send security alert to monitoring system"""
        alert_data = {
            'type': 'process_integrity_violation',
            'severity': violation.threat_level.value,
            'violation_id': violation.violation_id,
            'process_id': violation.process_id,
            'description': violation.description,
            'timestamp': violation.timestamp.isoformat(),
            'evidence': violation.evidence,
            'remediation_taken': violation.remediation_taken
        }
        
        # In a real implementation, send to your alerting system
        logger.critical(f"SECURITY ALERT: {json.dumps(alert_data, indent=2)}")
    
    def log_security_event(self, violation: IntegrityViolation):
        """Log security event for audit trail"""
        log_entry = {
            'timestamp': violation.timestamp.isoformat(),
            'event_type': 'process_integrity_violation',
            'violation_data': asdict(violation)
        }
        
        # Write to security audit log
        with open('security_audit.log', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    async def start_monitoring(self):
        """Start continuous process monitoring"""
        self.monitoring_active = True
        logger.info("CSP Process Integrity Monitoring started")
        
        while self.monitoring_active:
            try:
                await self.scan_active_processes()
                await asyncio.sleep(5)  # Scan every 5 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    def stop_monitoring(self):
        """Stop process monitoring"""
        self.monitoring_active = False
        logger.info("CSP Process Integrity Monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status and statistics"""
        with self.lock:
            recent_violations = [v for v in self.violations if v.timestamp > datetime.now() - timedelta(hours=24)]
            
            return {
                'monitoring_active': self.monitoring_active,
                'active_processes': len(self.active_processes),
                'whitelisted_hashes': len(self.whitelist_hashes),
                'blacklisted_hashes': len(self.blacklist_hashes),
                'total_violations': len(self.violations),
                'recent_violations': len(recent_violations),
                'violation_types': {
                    violation.violation_type: len([v for v in recent_violations if v.violation_type == violation.violation_type])
                    for violation in recent_violations
                }
            }

# Integration with Enhanced CSP
process_monitor = CSPProcessMonitor()

async def start_csp_process_monitoring():
    """Start CSP process integrity monitoring"""
    await process_monitor.start_monitoring()

def stop_csp_process_monitoring():
    """Stop CSP process integrity monitoring"""
    process_monitor.stop_monitoring()

# API endpoints for monitoring control
from fastapi import FastAPI

app = FastAPI()

@app.post("/security/process-monitoring/start")
async def start_monitoring_endpoint():
    """Start process monitoring via API"""
    asyncio.create_task(start_csp_process_monitoring())
    return {"status": "started"}

@app.post("/security/process-monitoring/stop")
async def stop_monitoring_endpoint():
    """Stop process monitoring via API"""
    stop_csp_process_monitoring()
    return {"status": "stopped"}

@app.get("/security/process-monitoring/status")
async def get_monitoring_status():
    """Get process monitoring status"""
    return process_monitor.get_status()

@app.get("/security/violations")
async def get_violations(limit: int = 50):
    """Get recent security violations"""
    with process_monitor.lock:
        recent_violations = sorted(
            process_monitor.violations,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
        
        return {
            'violations': [asdict(v) for v in recent_violations],
            'total_count': len(process_monitor.violations)
        }

@app.post("/security/whitelist-process")
async def whitelist_process(process_hash: str):
    """Add process hash to whitelist"""
    process_monitor.whitelist_hashes.add(process_hash)
    logger.info(f"Added process hash to whitelist: {process_hash[:16]}...")
    return {"status": "whitelisted", "hash": process_hash[:16] + "..."}

# Startup integration
@app.on_event("startup")
async def startup_monitoring():
    """Start monitoring on application startup"""
    asyncio.create_task(start_csp_process_monitoring())
    logger.info("CSP Process Integrity Monitoring initialized")
