import motor.motor_asyncio
from beanie import init_beanie
from app.models import User, UserDetails
from app.core import Settings

async def init_db():
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(Settings.MONGODB_URI)
        database = client[Settings.MONGODB_DB_NAME]
        await init_beanie(database, document_models=[User, UserDetails])
    except Exception as e:
        raise RuntimeError("Database initialization failed") from e