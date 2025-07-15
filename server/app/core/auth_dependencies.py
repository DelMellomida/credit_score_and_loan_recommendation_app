# app/auth_dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.core.security import decode_token
from app.services.auth_service import auth_service
from typing import Dict

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    """
    Dependency to get current authenticated user from JWT token.
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        Dict: User information
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the JWT token
        payload = decode_token(token)
        if payload is None:
            raise credentials_exception
        
        # Extract email from token payload
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
            
    except Exception:
        raise credentials_exception
    
    # Get user from database
    user = await auth_service.get_user_by_email(email)
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(current_user: Dict = Depends(get_current_user)) -> Dict:
    """
    Dependency to get current active user.
    Can be extended to check user status, permissions, etc.
    
    Args:
        current_user: User dict from get_current_user dependency
        
    Returns:
        Dict: Active user information
        
    Raises:
        HTTPException: If user is inactive
    """
    # You can add additional checks here
    # For example, check if user is active, has required permissions, etc.
    
    # Example: Check if user is active (if you have this field)
    # if not current_user.get("is_active", True):
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Inactive user"
    #     )
    
    return current_user

async def get_admin_user(current_user: Dict = Depends(get_current_user)) -> Dict:
    """
    Dependency to get current user with admin privileges.
    Example of role-based access control.
    
    Args:
        current_user: User dict from get_current_user dependency
        
    Returns:
        Dict: Admin user information
        
    Raises:
        HTTPException: If user is not an admin
    """
    # Example admin check (you would need to add role field to your User model)
    # if current_user.get("role") != "admin":
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Admin access required"
    #     )
    
    return current_user