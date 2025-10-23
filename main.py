import os
from dotenv import load_dotenv
from fastapi import FastAPI, status, Query, Header, HTTPException
from fastapi.responses import JSONResponse
from fetcher import fetch_311_data, cache_clear, cache_stats
from app_types import Borough

load_dotenv()  # ensure .env is loaded here too

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

app = FastAPI()

def _auth(x_admin_token: str | None):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN not configured")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/")
def read_root():
    return JSONResponse(
        content={"status": "ok", "message": "Server is healthy"},
        status_code=status.HTTP_200_OK
    )

@app.get("/fetch")
def get_311_data(limit: int = 10):
    data = fetch_311_data(limit)
    return {"count": len(data), "results": data}

@app.get("/fetch/{borough}")
def fetch_by_borough(borough: Borough, limit: int = Query(10, ge=1, le=1000)):
    borough_param = "STATEN ISLAND" if borough == Borough.STATEN_ISLAND else borough.value.upper()
    data = fetch_311_data(limit, borough=borough_param)
    return {"borough": borough_param, "count": len(data), "results": data}

@app.post("/admin/cache/clear")
def admin_cache_clear(x_admin_token: str | None = Header(default=None)):
    _auth(x_admin_token)
    cache_clear()
    return {"ok": True}

@app.get("/admin/cache/stats")
def admin_cache_stats(x_admin_token: str | None = Header(default=None)):
    _auth(x_admin_token)
    return cache_stats()

@app.get("/health")
def health():
    """Lightweight health check with cache stats."""
    return {
        "status": "ok",
        "service": "nyc-311",
    }