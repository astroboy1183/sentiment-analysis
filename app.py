from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentiment_service import create_service, SentimentService

app = FastAPI(title="Sentiment Analysis API", version="0.1.0")
sentiment_service: SentimentService = create_service()


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    sentiment: str
    score: float
    scores: dict


@app.get("/")
def root() -> dict:
    return {"message": "Welcome to the Sentiment Analysis API"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = sentiment_service.analyze(request.text)
    except ValueError as exc:  # convert validation error to HTTP 400
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnalyzeResponse(
        sentiment=result["label"],
        score=result["score"],
        scores=result["scores"],
    )
