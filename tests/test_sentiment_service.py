from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentiment_service import SentimentService


@pytest.fixture()
def service() -> SentimentService:
    return SentimentService()


def test_positive_sentiment(service: SentimentService) -> None:
    result = service.analyze("I absolutely love this product!")
    assert result["label"] == "positive"
    assert result["score"] > 0.05


def test_negative_sentiment(service: SentimentService) -> None:
    result = service.analyze("This is the worst experience I've ever had.")
    assert result["label"] == "negative"
    assert result["score"] < -0.05


def test_neutral_sentiment(service: SentimentService) -> None:
    result = service.analyze("The product is okay and does its job.")
    assert result["label"] == "neutral"
    assert -0.05 < result["score"] < 0.05


def test_rejects_empty_text(service: SentimentService) -> None:
    with pytest.raises(ValueError):
        service.analyze("   ")


def test_rejects_none(service: SentimentService) -> None:
    with pytest.raises(ValueError):
        service.analyze(None)  # type: ignore[arg-type]
