# Sentiment Analysis

A minimal sentiment analysis application built with FastAPI and a lightweight rule-based classifier. It exposes a REST endpoint for analyzing text and a reusable service class for programmatic use.

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the API

Start the FastAPI server with Uvicorn:

```bash
uvicorn app:app --reload
```

Visit `http://127.0.0.1:8000/docs` to explore the interactive API documentation. Example request:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
    -H "Content-Type: application/json" \
    -d '{"text": "This library is fantastic!"}'
```

## Testing

Run the automated tests with pytest:

```bash
pytest
```
