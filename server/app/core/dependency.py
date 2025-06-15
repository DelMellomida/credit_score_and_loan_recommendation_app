from fastapi import Depends, HTTPException, status, Request
from app.core.security import decode_access_token
from app.models import User
from app.schemas.user import UserOut
from app.exceptions import DatabaseException, DatabaseCreateException, DatabaseReadException, DatabaseUpdateException, DatabaseDeleteException
from app.exceptions import ProcessingException

# app/core/dependency.py
async def get_current_user(request: Request) -> User:
    # Get token from cookie
    token = request.cookies.get("access_token")
    if not token:
        raise ProcessingException(f"Not authenticated - no token found")
    # Clean the token
    token = token.strip('"') 
    if token.startswith('Bearer '):
        token = token[7:] 
    
    try:
        payload = decode_access_token(token)
        if not payload or "sub" not in payload:
            raise ProcessingException(f"Invalid or expired token")
        
        user_id = payload["sub"]
        user = await User.get(user_id)
        if not user:
            raise DatabaseReadException(f"Database read failed: User not found")
        return user
        
    except Exception as e:
        raise ProcessingException(f"Authentication failed: {str(e)}")

# Debug version to see what's happening
async def get_current_user_debug(request: Request) -> User:
    # Get token from cookie
    token = request.cookies.get("access_token")
    print(f"Raw token from cookie: {repr(token)}")
    
    if not token:
        raise ProcessingException(f"Not authenticated - no token found")
    
    # Handle different cookie formats
    original_token = token
    if token.startswith('"Bearer ') and token.endswith('"'):
        # Remove quotes and "Bearer " prefix: "Bearer eyJ..." -> eyJ...
        token = token[8:-1]  # Remove first 8 chars ("Bearer ) and last char (")
        print(f"Cleaned token (format 1): {token[:20]}...")
    elif token.startswith('Bearer '):
        # Remove "Bearer " prefix: Bearer eyJ... -> eyJ...
        token = token[7:]
        print(f"Cleaned token (format 2): {token[:20]}...")
    elif token.startswith('"') and token.endswith('"'):
        # Remove quotes: "eyJ..." -> eyJ...
        token = token[1:-1]
        print(f"Cleaned token (format 3): {token[:20]}...")
    else:
        print(f"Token used as-is: {token[:20]}...")
    
    payload = decode_access_token(token)
    print(f"Decoded payload: {payload}")
    
    if payload is None or "sub" not in payload:
        raise ProcessingException(f"Invalid or expired token")
    
    user_id = payload["sub"]
    print(f"Looking for user with ID: {user_id}")
    
    user = await User.get(user_id)
    if user is None:
        print(f"User not found with ID: {user_id}")
        raise DatabaseReadException(f"Database read failed: User not found")
    
    print(f"Found user: {user.email} (ID: {user.id})")
    return user

async def get_current_user_response(request: Request) -> UserOut:
    user = await get_current_user(request)
    return UserOut(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at
    )