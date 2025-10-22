from fastapi import FastAPI, status, Query
from fastapi.responses import JSONResponse
from fetcher import fetch_311_data
from app_types import Borough

app = FastAPI()


@app.get("/")
def read_root():
    return JSONResponse(
        content={"status": "ok", "message": "Server is healthy"},
        status_code=status.HTTP_200_OK
    )


@app.get("/fetch")
def get_311_data(limit: int = 10):
    data = fetch_311_data(limit)
    return {
        "count": len(data),
        "results": data
    }


@app.get("/fetch/{borough}")
def fetch_by_borough(
    borough: Borough,
    limit: int = Query(10, ge=1, le=1000)
):
    borough_param = "STATEN ISLAND" if borough == Borough.STATEN_ISLAND else borough.value.upper()
    data = fetch_311_data(limit, borough=borough_param)
    return {
        "borough": borough_param,
        "count": len(data),
        "results": data
    }

@app.get("/health")
def health():
    """Lightweight health check with cache stats."""
    return {
        "status": "ok",
        "service": "nyc-311",
    }