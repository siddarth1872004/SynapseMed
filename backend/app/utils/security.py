import os
import uuid
from pathlib import Path
from fastapi import HTTPException, status
from app.config import settings

def sanitize_filename(filename: str) -> str:
    """
    Sanitize the input filename by taking only the basename to strip traversal paths.
    """
    if not filename:
         raise HTTPException(
             status_code=status.HTTP_400_BAD_REQUEST,
             detail="Filename cannot be empty"
         )
    # Strip path traversal sequences using os.path.basename
    safe_name = os.path.basename(filename)
    # Fallback to random if basename resolved to empty
    if not safe_name or safe_name in {".", ".."}:
        safe_name = f"upload_{uuid.uuid4().hex}"
    return safe_name

def validate_file_size(file_size: int) -> None:
    """
    Check if file size exceeds our limits.
    """
    if file_size > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_BYTES / (1024*1024):.1f} MB."
        )

def validate_magic_bytes(content: bytes, filename: str) -> str:
    """
    Inspect magic bytes of uploaded files to verify they are legitimate PDF, JPEG, PNG, TIFF or plain text.
    Returns the resolved extension.
    """
    ext = filename.split('.')[-1].lower() if '.' in filename else ''
    
    # Check Magic Bytes
    if content.startswith(b'%PDF'):
        if ext != 'pdf':
             raise HTTPException(status_code=400, detail="File content mismatch: Expected PDF")
        return 'pdf'
    elif content.startswith(b'\x89PNG\r\n\x1a\n'):
        if ext != 'png':
             raise HTTPException(status_code=400, detail="File content mismatch: Expected PNG")
        return 'png'
    elif content.startswith(b'\xff\xd8\xff'):
        if ext not in {'jpg', 'jpeg'}:
             raise HTTPException(status_code=400, detail="File content mismatch: Expected JPEG")
        return ext
    elif content.startswith(b'II*\x00') or content.startswith(b'MM\x00*'):
        if ext not in {'tif', 'tiff'}:
             raise HTTPException(status_code=400, detail="File content mismatch: Expected TIFF")
        return ext
    
    # Try decoding for text files
    if ext == 'txt':
        try:
            content.decode('utf-8')
            return 'txt'
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File content mismatch: Text file contains invalid binary content")
            
    # For docx, standard zip structure starts with PK\x03\x04
    elif content.startswith(b'PK\x03\x04') and ext == 'docx':
        return 'docx'
        
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unsupported or invalid file format. Allowed: PDF, PNG, JPG/JPEG, TIFF, DOCX, TXT."
    )

def secure_save_file(content: bytes, original_filename: str, upload_dir: Path) -> Path:
    """
    Securely saves file with size verification, path traversal checks, random name assignment,
    and strict directory boundary checks.
    """
    # Verify file size
    validate_file_size(len(content))
    
    # Sanitize and validate magic bytes
    safe_orig = sanitize_filename(original_filename)
    validated_ext = validate_magic_bytes(content, safe_orig)
    
    # Generate unique, random filename to prevent overwriting or predictable URLs
    unique_name = f"{uuid.uuid4().hex}.{validated_ext}"
    
    # Strict boundary check
    target_path = Path(upload_dir) / unique_name
    
    # Resolve absolute paths
    resolved_target = target_path.resolve()
    resolved_sandbox = Path(upload_dir).resolve()
    
    # Enforce directory boundary verification to prevent partial matches / escape
    # Ensure sandbox has trailing slash or check starts with sandbox path + separator
    if not str(resolved_target).startswith(str(resolved_sandbox) + os.sep):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File path resolves outside sandbox directory boundary"
        )
        
    # Write the file content
    try:
        resolved_target.write_bytes(content)
        # Ensure it has restrictive permissions (chmod 600 - user read/write only, not executable)
        os.chmod(resolved_target, 0o600)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to securely save upload file: {str(e)}"
        )
        
    return resolved_target
