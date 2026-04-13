import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.router import router as chat_router
from service.swin_service import get_model as get_swin
from service.efficient_service import get_model as get_eff

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_swin()          # load SwinV2
    get_eff()           # load EfficientNet
    yield

app = FastAPI(title="Fundus DR Grading API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api", tags=["Predict"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)