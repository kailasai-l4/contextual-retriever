import os
from typing import List

SUPPORTED_EXTENSIONS = [".txt", ".md", ".json", ".csv"]

def find_supported_files(directory: str) -> List[str]:
    """Recursively find all supported files in the directory and subdirectories."""
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                file_list.append(os.path.join(root, file))
    return file_list
