import requests
import time

API_URL = "http://localhost:8001/process/"
PROGRESS_URL = "http://localhost:8001/process/ingest-progress/"
FILE_PATH = "README.md"
COLLECTION_NAME = "test"

def test_single_file_upload():
    with open(FILE_PATH, "rb") as f:
        files = {"file": (FILE_PATH, f)}
        data = {"collection_name": COLLECTION_NAME}
        response = requests.post(API_URL, files=files, data=data)
        print("Status Code:", response.status_code)
        print("Response:", response.text)
        assert response.status_code == 200, f"Unexpected status: {response.status_code}"
        resp_json = response.json()
        assert resp_json.get("status") == "started", "Upload did not start correctly"
        task_id = resp_json.get("task_id")
        assert task_id, "No task_id returned"
        # Poll progress
        for _ in range(10):
            progress_resp = requests.get(PROGRESS_URL + task_id)
            if progress_resp.status_code == 200:
                progress = progress_resp.json()
                print("Progress:", progress)
                if progress.get("done"):
                    assert progress.get("processed") == progress.get("total"), "Not all chunks processed"
                    print("Ingestion complete!")
                    break
            time.sleep(1)
        else:
            assert False, "Ingestion did not complete in time"

if __name__ == "__main__":
    test_single_file_upload()