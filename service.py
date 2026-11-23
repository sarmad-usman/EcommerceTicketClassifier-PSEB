import json, os, re
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATEGORIES = [
    "Billing and Payment", "Order Status", "Delivery Issues", "Returns and Refunds",
    "Account Access", "Technical Issues", "Product Information", "Cancellation Requests",
    "Damaged or Missing Items", "Other Inquiries"
]
Category = Literal[
    "Billing and Payment", "Order Status", "Delivery Issues", "Returns and Refunds",
    "Account Access", "Technical Issues", "Product Information", "Cancellation Requests",
    "Damaged or Missing Items", "Other Inquiries"
]

class TicketRequest(BaseModel):
    ticket: str = Field(min_length=1, max_length=5000)
    @field_validator("ticket")
    @classmethod
    def non_blank(cls, value):
        if not value.strip():
            raise ValueError("Ticket must not be blank.")
        return value.strip()

class TicketResult(BaseModel):
    category: Category
    summary: str = Field(min_length=5, max_length=300)
    mode: str = "mock"
    prompt_version: str = "v2"

    @field_validator("summary")
    @classmethod
    def summary_not_blank(cls, value):
        if not value.strip():
            raise ValueError("Summary must not be blank.")
        return value.strip()

DATA_PATH = Path(__file__).parent / "data" / "tickets.csv"

def _training_data():
    df = pd.read_csv(DATA_PATH)
    return df[df["split"] == "development"]

def _tfidf_predict(ticket: str):
    train = _training_data()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(train["ticket"].tolist() + [ticket])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    best = int(scores.argmax())
    return str(train.iloc[best]["expected_category"]), float(scores[best])

def _keyword_predict(ticket: str):
    t = ticket.lower()
    rules = [
        ("Billing and Payment", ["payment", "charged", "charge", "invoice", "receipt", "bank", "card", "fee", "tax", "paid"]),
        ("Order Status", ["order status", "order confirmation", "processing", "purchase history", "dispatch", "accepted"]),
        ("Delivery Issues", ["delivery", "delivered", "courier", "tracking", "parcel", "shipment", "in transit", "late"]),
        ("Returns and Refunds", ["return", "refund", "store credit", "return label"]),
        ("Account Access", ["login", "log in", "sign in", "password", "account", "verification", "username", "two-factor"]),
        ("Technical Issues", ["app", "application", "website", "error", "crash", "freeze", "loading", "page", "503", "cart"]),
        ("Product Information", ["size", "colour", "color", "material", "dimensions", "warranty", "specification", "compatibility", "product"]),
        ("Cancellation Requests", ["cancel", "cancellation", "stop an order"]),
        ("Damaged or Missing Items", ["damaged", "broken", "crack", "missing", "defective", "wrong item", "empty package", "scratches"]),
    ]
    for category, words in rules:
        if any(word in t for word in words):
            return category
    return "Other Inquiries"

def _summary(ticket: str):
    clean = re.sub(r"\s+", " ", ticket).strip()
    # Keep a factual extract; do not invent a resolution.
    if len(clean) > 180:
        clean = clean[:177].rsplit(" ", 1)[0] + "..."
    return clean

def analyze_ticket(ticket: str, prompt_version: str = "v2", mode: str | None = None):
    selected_mode = (mode or os.getenv("LLM_MODE", "mock")).lower()
    if selected_mode not in ("mock", "live"):
        raise ValueError("LLM_MODE must be 'mock' or 'live'.")
    if selected_mode == "live":
        return _live_analyze(ticket, prompt_version)
    if prompt_version == "v1":
        category = _keyword_predict(ticket)
    elif prompt_version == "v2":
        category, _ = _tfidf_predict(ticket)
    else:
        raise ValueError("prompt_version must be v1 or v2")
    return TicketResult(category=category, summary=_summary(ticket), mode="mock", prompt_version=prompt_version)

def _live_analyze(ticket: str, prompt_version: str):
    from openai import OpenAI
    from pathlib import Path
    prompt_file = Path(__file__).parent / "prompts" / f"prompt_{prompt_version}.txt"
    if not prompt_file.exists():
        raise ValueError("prompt_version must be v1 or v2")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not api_key or not model:
        raise RuntimeError("Live mode requires OPENAI_API_KEY and LLM_MODEL in backend environment.")
    client = OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL") or None)
    system_prompt = prompt_file.read_text(encoding="utf-8")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Follow the classification instructions. Return JSON only."},
            {"role": "user", "content": system_prompt.format(ticket=ticket)}
        ],
        temperature=0,
        max_tokens=180,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    if not raw:
        raise RuntimeError("Model returned an empty response.")
    data = json.loads(raw)
    return TicketResult.model_validate({**data, "mode": "live", "prompt_version": prompt_version})
