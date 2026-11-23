from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    assert client.get("/health").status_code == 200

def test_valid_ticket():
    response = client.post("/analyze", json={"ticket": "I was charged twice for the same order."})
    assert response.status_code == 200
    body = response.json()
    assert body["category"] in [
        "Billing and Payment", "Order Status", "Delivery Issues", "Returns and Refunds",
        "Account Access", "Technical Issues", "Product Information", "Cancellation Requests",
        "Damaged or Missing Items", "Other Inquiries"
    ]
    assert body["summary"]

def test_blank_ticket_rejected():
    response = client.post("/analyze", json={"ticket": "   "})
    assert response.status_code == 422

def test_non_string_ticket_rejected():
    response = client.post("/analyze", json={"ticket": 123})
    assert response.status_code == 422
