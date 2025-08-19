"""
import time
Secure File Upload Handler for A2A Platform
Provides comprehensive security validation for file uploads
"""

import os
import magic
import hashlib
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, BinaryIO, Tuple
from datetime import datetime
from enum import Enum

from fastapi import UploadFile, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FileValidationResult(str, Enum):
    """File validation results"""
    APPROVED = "approved"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class SecurityThreat(str, Enum):
    """Security threat types"""
    MALWARE = "malware"
    EXECUTABLE = "executable"  
    OVERSIZED = "oversized"
    FORBIDDEN_TYPE = "forbidden_type"
    SUSPICIOUS_NAME = "suspicious_name"
    CORRUPTED = "corrupted"


class FileUploadConfig(BaseModel):
    """Configuration for file upload security"""
    max_file_size: int = 10 * 1024 * 1024  # 10MB default
    allowed_extensions: List[str] = ['.txt', '.json', '.csv', '.xml', '.pdf', '.jpg', '.png']
    allowed_mime_types: List[str] = [
        'text/plain', 'text/csv', 'application/json', 'application/xml',
        'application/pdf', 'image/jpeg', 'image/png', 'image/gif'
    ]
    quarantine_path: str = "/tmp/quarantine"
    scan_enabled: bool = True
    deep_inspection: bool = True
    virus_scan: bool = False  # Requires external scanner


class SecureFileValidator:
    """Comprehensive file security validator"""
    
    def __init__(self, config: FileUploadConfig):
        self.config = config
        self.quarantine_dir = Path(config.quarantine_path)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file type detection
        try:
            self.mime_detector = magic.Magic(mime=True)
            self.magic_available = True
        except Exception as e:
            logger.warning(f"Magic file type detection not available: {e}")
            self.magic_available = False
    
    async def validate_upload(self, file: UploadFile) -> Dict[str, any]:
        """Perform comprehensive file validation"""
        validation_start = datetime.utcnow()
        threats_detected = []
        
        try:
            # Basic metadata validation
            filename_validation = self._validate_filename(file.filename)
            if filename_validation["threats"]:
                threats_detected.extend(filename_validation["threats"])
            
            # Read file content for inspection
            content = await file.read()
            file_size = len(content)
            
            # Reset file position for later use
            await file.seek(0)
            
            # Size validation
            if file_size > self.config.max_file_size:
                threats_detected.append({
                    "type": SecurityThreat.OVERSIZED,
                    "description": f"File size {file_size} exceeds limit {self.config.max_file_size}",
                    "severity": "high"
                })
            
            # MIME type detection and validation
            mime_validation = self._validate_mime_type(content, file.content_type)
            if mime_validation["threats"]:
                threats_detected.extend(mime_validation["threats"])
            
            # Deep content inspection
            if self.config.deep_inspection:
                content_validation = self._inspect_file_content(content, filename_validation["safe_name"])
                if content_validation["threats"]:
                    threats_detected.extend(content_validation["threats"])
            
            # File signature validation
            signature_validation = self._validate_file_signature(content)
            if signature_validation["threats"]:
                threats_detected.extend(signature_validation["threats"])
            
            # Determine overall result
            high_severity_threats = [t for t in threats_detected if t.get("severity") == "high"]
            
            if high_severity_threats:
                result = FileValidationResult.REJECTED
                # Quarantine suspicious files for analysis
                quarantine_path = await self._quarantine_file(file, content, threats_detected)
            elif threats_detected:
                result = FileValidationResult.QUARANTINED
                quarantine_path = await self._quarantine_file(file, content, threats_detected)
            else:
                result = FileValidationResult.APPROVED
                quarantine_path = None
            
            # Generate file hash for tracking
            file_hash = hashlib.sha256(content).hexdigest()
            
            validation_result = {
                "result": result,
                "filename": filename_validation["safe_name"],
                "original_filename": file.filename,
                "file_size": file_size,
                "mime_type": mime_validation["detected_type"],
                "file_hash": file_hash,
                "threats": threats_detected,
                "quarantine_path": quarantine_path,
                "validation_time": (datetime.utcnow() - validation_start).total_seconds(),
                "timestamp": validation_start.isoformat()
            }
            
            # Log validation result
            logger.info(f"File validation completed: {file.filename} -> {result}")
            if threats_detected:
                logger.warning(f"Threats detected in {file.filename}: {len(threats_detected)} issues")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="File validation failed"
            )
    
    def _validate_filename(self, filename: str) -> Dict[str, any]:
        """Validate filename for security threats"""
        if not filename:
            return {
                "safe_name": "unnamed_file",
                "threats": [{
                    "type": SecurityThreat.SUSPICIOUS_NAME,
                    "description": "No filename provided",
                    "severity": "medium"
                }]
            }
        
        threats = []
        
        # Check for path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            threats.append({
                "type": SecurityThreat.SUSPICIOUS_NAME,
                "description": "Path traversal attempt in filename",
                "severity": "high"
            })
        
        # Check for suspicious characters
        suspicious_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
        if any(char in filename for char in suspicious_chars):
            threats.append({
                "type": SecurityThreat.SUSPICIOUS_NAME,
                "description": "Suspicious characters in filename",
                "severity": "medium"
            })
        
        # Check extension
        file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        if file_ext and file_ext not in self.config.allowed_extensions:
            threats.append({
                "type": SecurityThreat.FORBIDDEN_TYPE,
                "description": f"File extension {file_ext} not allowed",
                "severity": "high"
            })
        
        # Check for executable extensions
        dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.scr', '.vbs', '.js', '.jar',
            '.sh', '.py', '.pl', '.php', '.asp', '.jsp'
        }
        if file_ext in dangerous_extensions:
            threats.append({
                "type": SecurityThreat.EXECUTABLE,
                "description": f"Executable file extension detected: {file_ext}",
                "severity": "high"
            })
        
        # Create safe filename
        safe_name = "".join(c for c in filename if c.isalnum() or c in '.-_')[:100]
        if not safe_name:
            safe_name = f"file_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "safe_name": safe_name,
            "threats": threats
        }
    
    def _validate_mime_type(self, content: bytes, declared_type: str) -> Dict[str, any]:
        """Validate MIME type against content"""
        threats = []
        detected_type = declared_type
        
        # Detect actual MIME type from content
        if self.magic_available and content:
            try:
                detected_type = self.mime_detector.from_buffer(content)
            except Exception as e:
                logger.warning(f"MIME type detection failed: {e}")
        
        # Check if detected type matches declared type
        if detected_type != declared_type:
            threats.append({
                "type": SecurityThreat.SUSPICIOUS_NAME,
                "description": f"MIME type mismatch: declared {declared_type}, detected {detected_type}",
                "severity": "medium"
            })
        
        # Check if type is allowed
        if detected_type not in self.config.allowed_mime_types:
            threats.append({
                "type": SecurityThreat.FORBIDDEN_TYPE,
                "description": f"MIME type {detected_type} not allowed",
                "severity": "high"
            })
        
        return {
            "detected_type": detected_type,
            "threats": threats
        }
    
    def _inspect_file_content(self, content: bytes, filename: str) -> Dict[str, any]:
        """Deep inspection of file content"""
        threats = []
        
        try:
            # Check for embedded executables (PE headers, ELF headers)
            executable_signatures = [
                b'MZ',      # PE executable
                b'\x7fELF', # ELF executable  
                b'\xca\xfe\xba\xbe',  # Java class file
                b'PK\x03\x04',        # ZIP/JAR (could contain executables)
            ]
            
            for signature in executable_signatures:
                if content.startswith(signature):
                    threats.append({
                        "type": SecurityThreat.EXECUTABLE,
                        "description": f"Executable signature detected: {signature.hex()}",
                        "severity": "high"
                    })
            
            # Check for script content in non-script files
            if filename.endswith(('.txt', '.csv', '.json')):
                script_patterns = [
                    b'<script', b'javascript:', b'eval(', b'exec(',
                    b'system(', b'shell_exec', b'passthru', b'<?php'
                ]
                
                content_lower = content.lower()
                for pattern in script_patterns:
                    if pattern in content_lower:
                        threats.append({
                            "type": SecurityThreat.MALWARE,
                            "description": f"Suspicious script pattern found: {pattern.decode('utf-8', errors='ignore')}",
                            "severity": "high"
                        })
            
            # Check for abnormally high entropy (possible encryption/obfuscation)
            if len(content) > 1000:
                entropy = self._calculate_entropy(content)
                if entropy > 7.5:  # High entropy threshold
                    threats.append({
                        "type": SecurityThreat.SUSPICIOUS_NAME,
                        "description": f"High entropy detected: {entropy:.2f} (possible obfuscation)",
                        "severity": "medium"
                    })
            
        except Exception as e:
            logger.error(f"Content inspection error: {e}")
            threats.append({
                "type": SecurityThreat.CORRUPTED,
                "description": "File content inspection failed",
                "severity": "medium"
            })
        
        return {"threats": threats}
    
    def _validate_file_signature(self, content: bytes) -> Dict[str, any]:
        """Validate file signature matches extension"""
        threats = []
        
        # Common file signatures
        signatures = {
            'pdf': [b'%PDF'],
            'jpg': [b'\xff\xd8\xff'],
            'png': [b'\x89PNG\r\n\x1a\n'],
            'gif': [b'GIF87a', b'GIF89a'],
            'zip': [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08']
        }
        
        # Check if content has valid signature for claimed type
        if len(content) >= 8:  # Minimum bytes for signature check
            signature_found = False
            for file_type, sigs in signatures.items():
                for sig in sigs:
                    if content.startswith(sig):
                        signature_found = True
                        break
                if signature_found:
                    break
            
            # If it looks like a known binary format but no signature matches
            if not signature_found and any(b in content[:100] for b in [b'\x00', b'\xff', b'\xfe']):
                threats.append({
                    "type": SecurityThreat.SUSPICIOUS_NAME,
                    "description": "Binary content without recognizable file signature",
                    "severity": "medium"
                })
        
        return {"threats": threats}
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0
        
        # Count frequency of each byte
        freq = {}
        for byte in data:
            freq[byte] = freq.get(byte, 0) + 1
        
        # Calculate entropy
        entropy = 0
        data_len = len(data)
        for count in freq.values():
            p = count / data_len
            if p > 0:
                entropy -= p * (p.bit_length() - 1)
        
        return entropy
    
    async def _quarantine_file(self, file: UploadFile, content: bytes, threats: List[Dict]) -> str:
        """Quarantine suspicious file"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            quarantine_filename = f"{timestamp}_{file.filename}"
            quarantine_path = self.quarantine_dir / quarantine_filename
            
            # Save file content
            with open(quarantine_path, 'wb') as f:
                f.write(content)
            
            # Save metadata
            metadata = {
                "original_filename": file.filename,
                "quarantine_time": datetime.utcnow().isoformat(),
                "threats": threats,
                "file_size": len(content),
                "file_hash": hashlib.sha256(content).hexdigest()
            }
            
            metadata_path = quarantine_path.with_suffix('.metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"File quarantined: {quarantine_path}")
            return str(quarantine_path)
            
        except Exception as e:
            logger.error(f"Quarantine failed: {e}")
            return None


class SecureFileUploadHandler:
    """Main handler for secure file uploads"""
    
    def __init__(self, config: FileUploadConfig = None):
        self.config = config or FileUploadConfig()
        self.validator = SecureFileValidator(self.config)
    
    async def handle_upload(self, file: UploadFile) -> Dict[str, any]:
        """Handle secure file upload with comprehensive validation"""
        
        # Validate file
        validation_result = await self.validator.validate_upload(file)
        
        # Handle based on validation result
        if validation_result["result"] == FileValidationResult.REJECTED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File rejected due to security threats: {[t['description'] for t in validation_result['threats']]}"
            )
        
        elif validation_result["result"] == FileValidationResult.QUARANTINED:
            logger.warning(f"File quarantined: {file.filename}")
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail="File quarantined for security review"
            )
        
        # File approved - return validation info
        return {
            "status": "approved",
            "filename": validation_result["filename"],
            "file_hash": validation_result["file_hash"],
            "size": validation_result["file_size"],
            "mime_type": validation_result["mime_type"]
        }


# Export main classes
__all__ = [
    'FileValidationResult',
    'SecurityThreat', 
    'FileUploadConfig',
    'SecureFileValidator',
    'SecureFileUploadHandler'
]