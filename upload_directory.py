import os
import requests
import time
from tqdm import tqdm

API_URL = "http://localhost:8000/process/"
PROGRESS_URL = "http://localhost:8000/process/ingest-progress/"
SUPPORTED_EXTENSIONS = {".md", ".txt", ".csv", ".json"}

def find_files(directory):
    """Recursively find all supported files in the directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, fname))
    return files

def upload_file(filepath, collection_name):
    with open(filepath, "rb") as f:
        files = {"file": (os.path.basename(filepath), f)}
        data = {"collection_name": collection_name}
        response = requests.post(API_URL, files=files, data=data)
        if response.status_code == 200:
            resp_json = response.json()
            task_id = resp_json.get("task_id")
            return task_id
        else:
            print(f"Failed to upload {filepath}: {response.text}")
            return None

def poll_progress(task_id):
    for _ in range(20):  # Wait up to 20 seconds per file
        resp = requests.get(PROGRESS_URL + task_id)
        if resp.status_code == 200:
            prog = resp.json()
            if prog.get("done"):
                return True
        time.sleep(1)
    return False

def upload_directory(directory, collection_name):
    files = find_files(directory)
    print(f"Found {len(files)} files to upload.")
    results = []
    for filepath in tqdm(files, desc="Uploading files", unit="file"):
        task_id = upload_file(filepath, collection_name)
        if task_id:
            success = poll_progress(task_id)
            results.append((filepath, success))
            tqdm.write(f"{'[OK]' if success else '[FAIL]'} {filepath}")
        else:
            results.append((filepath, False))
            tqdm.write(f"[FAIL] {filepath}")
    # Summary
    ok = sum(1 for _, s in results if s)
    print(f"\nUpload complete: {ok}/{len(files)} files succeeded.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload all files in a directory for ingestion.")
    parser.add_argument("directory", help="Path to the directory to upload from.")
    parser.add_argument("collection", help="Qdrant collection name to ingest into.")
    args = parser.parse_args()
    upload_directory(args.directory, args.collection)