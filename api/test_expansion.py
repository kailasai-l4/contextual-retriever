import requests
import json

API_URL = "http://localhost:8001/search"

def test_query_expansion():
    # Basic test with expansion enabled and model specified
    payload = {
        "query": "query expansion?",
        "use_expansion": True,
        "expansion_model": "openai",
        "limit": 2,
        "collection_name": "content_library"
    }
    resp = requests.post(API_URL, json=payload)
    print("/search with expansion status:", resp.status_code)
    data = resp.json()
    print(json.dumps(data, indent=2))
    assert resp.status_code == 200
    assert "expanded_query" in data
    assert data["expansion_model"] == "openai"
    assert isinstance(data["results"], list)
    print("Expansion test PASSED.")

if __name__ == "__main__":
    test_query_expansion()
