from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from app.schemas import UserCreate, UserLogin, UserOut
from app.services import AuthService
from app.core import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])

auth_service = AuthService()

@router.post('/register')
async def register_user(request: Request, user: UserCreate, response: Response):
    return await auth_service.register_user(user, response)

@router.post('/login')
async def login_user(request: Request, user: UserLogin, response: Response):
    return await auth_service.login_user(user, response)

@router.post('/logout')
async def logout_user(request: Request, response: Response):
    return await auth_service.logout_user(response)

@router.get('/me')
async def get_current_user(request: Request, user: UserOut = Depends(get_current_user)):
    return user