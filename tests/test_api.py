def test_health_endpoint_returns_200(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ("healthy", "degraded")

def test_classify_rejects_empty_text(api_client):
    response = api_client.post("/classify", json={"text": ""})
    assert response.status_code == 422

def test_classify_returns_recommendations(api_client):
    response = api_client.post("/classify", json={
        "text": "I feel hopeless and tired",
        "include_explanations": True
    })
    body = response.json()
    assert body["emotion"]
    assert len(body["recommendations"]) >= 1
    assert len(body["explanations"]) >= 1