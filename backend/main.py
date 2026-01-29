# backend/main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from causal_discovery import run_causal_discovery

app = FastAPI()

# Allow Streamlit frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use Streamlit Cloud domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/discover")
async def discover(request: Request):
    try:
        data = await request.json()
        df = pd.DataFrame(data)
        outputs = run_causal_discovery(df)
        return {"outputs": outputs}
    except Exception as e:
        return {"error": str(e)}
