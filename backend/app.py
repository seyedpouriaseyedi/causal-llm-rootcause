from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from causal_discovery import run_causal_discovery

app = FastAPI()

# CORS to allow Streamlit to call FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/discover")
async def discover(request: Request):
    data = await request.json()
    df = pd.DataFrame(data)
    paths = run_causal_discovery(df)
    return {"status": "ok", "outputs": paths}

