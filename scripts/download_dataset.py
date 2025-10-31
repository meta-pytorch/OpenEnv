# scripts/download_dataset.py
import requests
import os
import argparse

def download_file(url, local_filename):
    """Downloads a file from a given URL."""
    print(f"Downloading from: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Successfully saved to: {local_filename}")
    return local_filename

if __name__ == "__main__":
    # --- THIS IS THE NEW, FLEXIBLE PART ---
    parser = argparse.ArgumentParser(description="Download a dataset for the environment.")
    
    # The user must provide a URL with --url
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="The URL of the .jsonl dataset to download."
    )
    # The user specifies where to save the file with --output
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.jsonl",
        help="The local path to save the downloaded file."
    )
    args = parser.parse_args()
    
    # Run the download
    download_file(args.url, args.output)