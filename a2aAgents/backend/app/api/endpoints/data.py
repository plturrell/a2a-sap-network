from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_data():
    return {"message": "Data endpoint placeholder"}
