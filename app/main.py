from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.endpoints import processing, home

app = FastAPI()

# Configure CORS
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://127.0.0.1:5173"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
  
)

app.include_router(processing.router)
app.include_router(home.router)

