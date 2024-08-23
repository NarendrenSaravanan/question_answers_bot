from fastapi import APIRouter

from .bot import router as bot_router

main_router = APIRouter()

main_router.include_router(bot_router, prefix="/bot", tags=["bot"])

@main_router.get("/")
async def index():
    return {"message": "Hello From Zania!"}
