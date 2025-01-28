from config import redis_client
import requests
import json


def test_generate_image():
    base_url = "http://localhost:5432"
    test_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL2FpdGhld2F5LmNvbSIsImlhdCI6MTczODA3OTk5NCwiZXhwIjoxNzM4MTY2Mzk0LCJkYXRhIjp7InVzZXIiOnsiaWQiOjEsImVtYWlsIjoiZW5ndWVycmFuZWNrZXJ0QGlrLm1lIn19fQ.l8a6pf0k8NDc9g2pBQZ8b-8ripwc2IvsvETsu1DXqS8"  # Remplacez par un token valide

    # Test génération d'image
    response = requests.post(
        f"{base_url}/generate_image",
        params={"token": test_token},
        data={"prompt": "Un paysage montagneux"}
    )

    print("Test génération d'image:")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()


def test_history_retrieval():
    base_url = "http://localhost:5432"
    test_token = "votre_token_de_test"

    response = requests.get(
        f"{base_url}/api/history/generated",
        params={"token": test_token, "page": 1}
    )

    print("\nTest récupération historique:")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")