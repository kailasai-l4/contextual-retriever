import os
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Qdrant connection details from environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://135.181.129.60") # Use the IP directly as default
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None) # Optional API key

COLLECTION_NAME = "thondaimandala_kingdom"

print(f"Attempting to connect to Qdrant at {QDRANT_URL}:{QDRANT_PORT}")

try:
    # Initialize Qdrant client
    client = QdrantClient(
        url=QDRANT_URL,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        timeout=20 # Increased timeout
    )

    print(f"Checking collection: {COLLECTION_NAME}")

    # Check if collection exists
    collections_response = client.get_collections()
    collection_names = [c.name for c in collections_response.collections]

    if COLLECTION_NAME in collection_names:
        print(f"Collection '{COLLECTION_NAME}' exists.")
        
        # Get collection info
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection Info: {collection_info}")

        # Get point count
        count_response = client.count(collection_name=COLLECTION_NAME, exact=True)
        print(f"Point Count Response: {count_response}")
        print(f"Total points in '{COLLECTION_NAME}': {count_response.count}")

        # Try scrolling for a few points
        print(f"Attempting to scroll for up to 5 points...")
        try:
            scroll_response, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=5,
                with_payload=False, # Don't need payload for check
                with_vectors=False
            )
            print(f"Scroll response: Found {len(scroll_response)} points.")
            if scroll_response:
                point_ids = [p.id for p in scroll_response]
                print(f"Sample Point IDs: {point_ids}")
            else:
                print("Scroll returned no points.")

        except Exception as scroll_err:
            print(f"Error during scroll: {scroll_err}")

    else:
        print(f"Collection '{COLLECTION_NAME}' does NOT exist.")

except Exception as e:
    print(f"An error occurred: {e}") 