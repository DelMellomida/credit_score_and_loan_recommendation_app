from app.core import Settings, hash_password, verify_password, create_access_token
from app.models import User
from app.schemas import UserCreate, UserOut, UserLogin
from beanie import PydanticObjectId
from fastapi import HTTPException, status, Response
# import logging

class AuthService:
    async def register_user(self, user:UserCreate, response:Response):
        try:
            existing_user = await User.find_one(User.email == user.email)

            if existing_user:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User with this email already exists")
            
            hashed_pw = hash_password(user.password)
            new_user = User(
                name=user.name,
                email=user.email,
                phone=user.phone,
                password=hashed_pw,
                is_active=True
            )
            await new_user.save()

            access_token = create_access_token(
                data={"sub": str(new_user.id), "email": new_user.email}
            )
            response.set_cookie(
                key="access_token",
                value=f"Bearer {access_token}",
                httponly=True,
                secure=False,
                samesite="lax",
                max_age=Settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            )

            response.status_code = status.HTTP_201_CREATED
            return {"message": "User registered successfully", "redirect_url": "/dashboard"}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
        
    async def login_user(self, user:UserLogin, response:Response):
        try:
            existing_user = await User.find_one(User.email == user.email)
            if not existing_user:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid credentials")
            
            if not verify_password(user.password, existing_user.password):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email or password")
            
            access_token = create_access_token(
                data={"sub": str(existing_user.id), "email": existing_user.email}
            )
            response.set_cookie(
                key="access_token",
                value=f"Bearer {access_token}",
                httponly=True,
                secure=False,
                samesite="lax",
                max_age=Settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            )

            return {"message": "User logged in successfully", "redirect_url": "/dashboard"}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
        
    async def logout_user(self, response:Response):
        try:
            response.delete_cookie("access_token")
            return {"message": "User logged out successfully"}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))