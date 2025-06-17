from fastapi import Depends, HTTPException, status, Request
from app.core import decode_token
from app.schemas import UserOut
from app.models import User

async def get_current_user(request: Request) -> UserOut:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    token = token.split('"')
    if token.startswith("Bearer "):
        token = token[7:]

    try:
        payload = decode_token(token)
        if not payload or "sub" not in payload:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token or expired token")
        
        user_id = payload["sub"]
        user = await User.get(user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return UserOut.model_validate(user)
    except:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token or expired token")
    

