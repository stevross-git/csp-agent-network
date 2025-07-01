"""
File upload monitoring instrumentation
"""
import time
from typing import Optional
from fastapi import UploadFile

try:
    from monitoring import get_default
    monitor = get_default()
    MONITORING_ENABLED = True
except ImportError:
    monitor = None
    MONITORING_ENABLED = False

async def monitor_file_upload(
    file: UploadFile,
    file_type: Optional[str] = None
) -> dict:
    """Monitor file upload metrics"""
    if not MONITORING_ENABLED:
        return {}
    
    # Determine file type
    if not file_type:
        file_type = file.content_type or "unknown"
    
    # Get file size
    file_size = 0
    if file.file:
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
    
    # Record upload
    monitor.record_file_upload(file_type, file_size, True)
    
    return {
        "file_type": file_type,
        "file_size": file_size
    }

async def monitor_file_processing(
    file_type: str,
    operation: str,
    func,
    *args,
    **kwargs
):
    """Monitor file processing operations"""
    if not MONITORING_ENABLED:
        return await func(*args, **kwargs)
    
    start_time = time.time()
    try:
        result = await func(*args, **kwargs)
        return result
    finally:
        duration = time.time() - start_time
        monitor.record_file_processing(file_type, operation, duration)
