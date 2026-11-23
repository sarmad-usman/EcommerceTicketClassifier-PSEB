from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from service import TicketRequest, TicketResult, analyze_ticket

app = FastAPI(title="Ticket Classification Assistant", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=TicketResult)
def analyze(request: TicketRequest):
    try:
        return analyze_ticket(request.ticket, prompt_version="v2")
    except (RuntimeError, ValueError, ValidationError, Exception) as exc:
        raise HTTPException(status_code=502, detail=f"Analysis service unavailable or returned an invalid result: {str(exc)}")
