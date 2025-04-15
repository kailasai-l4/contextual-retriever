import requests

API_URL = "http://localhost:8001"

def test_health():
    resp = requests.get(f"{API_URL}/health")
    print("/health status:", resp.status_code, resp.json())
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"

def test_health_not_found():
    resp = requests.get(f"{API_URL}/not_a_real_endpoint")
    print("/not_a_real_endpoint status:", resp.status_code)
    assert resp.status_code == 404

def test_collections_crud():
    # Ensure clean state by deleting if exists
    resp = requests.delete(f"{API_URL}/collections/test_collection")
    # Ignore errors if it doesn't exist (404 is fine)
    
    # Create collection
    payload = {"collection_name": "test_collection", "vector_size": 32, "distance": "cosine"}
    resp = requests.post(f"{API_URL}/collections/", json=payload)
    print("/collections/ create status:", resp.status_code, resp.json())
    assert resp.status_code == 201
    assert resp.json()["status"] == "ok"

    # List collections
    resp = requests.get(f"{API_URL}/collections/")
    print("/collections/ list status:", resp.status_code, resp.json())
    assert resp.status_code == 200
    assert "test_collection" in [c['name'] if isinstance(c, dict) and 'name' in c else c for c in resp.json()["collections"]]

    # Get collection info
    resp = requests.get(f"{API_URL}/collections/test_collection")
    print("/collections/test_collection info status:", resp.status_code, resp.json())
    assert resp.status_code == 200
    assert "status" in resp.json() and isinstance(resp.json()["status"], str)

    # Delete collection
    resp = requests.delete(f"{API_URL}/collections/test_collection")
    print("/collections/test_collection delete status:", resp.status_code, resp.json())
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # Get deleted collection (should 404)
    resp = requests.get(f"{API_URL}/collections/test_collection")
    print("/collections/test_collection get after delete status:", resp.status_code)
    assert resp.status_code == 404

def test_search():
    payload = {"query": "test", "limit": 1, "use_expansion": False}
    resp = requests.post(f"{API_URL}/search/", json=payload)
    print("/search status:", resp.status_code, resp.json())
    assert resp.status_code == 200
    assert "results" in resp.json()

def test_search_in_collection():
    payload = {"query": "test", "limit": 1, "use_expansion": False, "collection_name": "content_library"}
    resp = requests.post(f"{API_URL}/search/", json=payload)
    print("/search in collection status:", resp.status_code, resp.json())
    assert resp.status_code == 200
    assert "results" in resp.json()

def test_search_invalid_payload():
    payload = {"limit": 1}
    resp = requests.post(f"{API_URL}/search/", json=payload)
    print("/search invalid payload status:", resp.status_code, resp.text)
    assert resp.status_code == 422

def test_search_internal_error():
    payload = {"query": None, "limit": 1, "use_expansion": False}
    resp = requests.post(f"{API_URL}/search/", json=payload)
    print("/search internal error status:", resp.status_code, resp.text)
    assert resp.status_code in (400, 422, 500)

if __name__ == "__main__":
    test_health()
    test_health_not_found()
    test_collections_crud()
    test_search()
    test_search_in_collection()
    test_search_invalid_payload()
    test_search_internal_error()
