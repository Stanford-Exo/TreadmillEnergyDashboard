#!/usr/bin/env python3
import os
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

URL_FILE = "data_urls.txt"
MAX_WORKERS = 8  # Adjust based on how aggressive you want to be (8-12 is ideal)
CHUNK_SIZE = 1024 * 1024  # 1MB memory buffer per write cycle

def download_file(url):
    """Downloads a single file tracking status and basic error handling."""
    url = url.strip()
    if not url:
        return None
    
    # Extract the filename from the end of the Stanford Stacks URL
    filename = url.split('/')[-1]
    
    # Spoof user-agent to look like a standard browser session
    req = Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    
    try:
        print(f"[STARTING] -> {filename}")
        with urlopen(req) as response, open(filename, 'wb') as out_file:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out_file.write(chunk)
        print(f"[SUCCESS]  -> {filename} saved.")
        return filename
    except HTTPError as e:
        print(f"[ERROR]    -> {filename} failed with HTTP Status: {e.code}")
    except URLError as e:
        print(f"[ERROR]    -> {filename} network connection failure: {e.reason}")
    except Exception as e:
        print(f"[ERROR]    -> {filename} encountered unexpected error: {e}")
    return None

def main():
    if not os.path.exists(URL_FILE):
        print(f"Error: Target tracking manifest '{URL_FILE}' not found in this folder.")
        sys.exit(1)
        
    with open(URL_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
        
    if not urls:
        print(f"Error: '{URL_FILE}' is empty.")
        sys.exit(1)

    print("=========================================")
    # Format 2026 current infrastructure details natively
    print(f" Launching Python Cluster Downloader ")
    print(f" Active Concurrency Pool: {MAX_WORKERS} Threads")
    print(f" Total Files Scheduled:   {len(urls)}")
    print("=========================================\n")

    # Leverage an asynchronous execution pool for parallel network I/O bound tasks
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all download tasks to the worker pool
        future_to_url = {executor.submit(download_file, url): url for url in urls}
        
        success_count = 0
        for future in as_completed(future_to_url):
            result = future.result()
            if result:
                success_count += 1

    print("\n=========================================")
    print(f" Execution Cycle Complete.")
    print(f" Successfully downloaded {success_count}/{len(urls)} files.")
    print("=========================================")

if __name__ == "__main__":
    main()