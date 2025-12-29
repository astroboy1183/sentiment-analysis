from __future__ import annotations

import re
from typing import Dict

_POSITIVE_WORDS = {
    "love",
    "amazing",
    "great",
    "good",
    "fantastic",
    "excellent",
    "enjoy",
    "happy",
    "wonderful",
    "awesome",
}

_NEGATIVE_WORDS = {
    "worst",
    "bad",
    "terrible",
    "awful",
    "hate",
    "disappointing",
    "poor",
    "sad",
    "horrible",
    "dreadful",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


class SentimentService:
    """Lightweight rule-based sentiment analysis service."""

    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze text and return sentiment label and scores.

        Args:
            text: Input string to analyze.

        Returns:
            A dictionary with the predicted label, compound score, and the
            positive/negative/neutral ratios observed in the text.

        Raises:
            ValueError: If ``text`` is empty or only contains whitespace.
        """

        if text is None:
            raise ValueError("Text cannot be None.")

        normalized = text.strip()
        if not normalized:
            raise ValueError("Text cannot be empty.")

        tokens = _tokenize(normalized)
        if not tokens:
            raise ValueError("Text must contain at least one word.")

        positive = sum(1 for token in tokens if token in _POSITIVE_WORDS)
        negative = sum(1 for token in tokens if token in _NEGATIVE_WORDS)
        total = len(tokens)
        neutral = total - positive - negative

        compound = (positive - negative) / total

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "score": compound,
            "scores": {
                "positive": positive / total,
                "negative": negative / total,
                "neutral": max(neutral, 0) / total,
            },
        }


def create_service() -> SentimentService:
    """Factory to create a SentimentService instance."""

    return SentimentService()
