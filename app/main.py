from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.endpoints import processing, home, analyzing

app = FastAPI()

# Configure CORS
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:5173"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
  
)

app.include_router(processing.router)
app.include_router(home.router)
app.include_router(analyzing.router)

