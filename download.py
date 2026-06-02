#!/usr/bin/env python3
import os
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

URL_FILE = "data_urls.txt"
MAX_WORKERS = 8  
CHUNK_SIZE = 1024 * 1024  

def download_file(url):
    url = url.strip()
    if not url:
        return None
        
    try:
        # Example URL: https://stacks.stanford.edu/file/druid:jj710vy7867/subject_01.zip
        # 1. Split by 'druid:' to isolate the ID and filename segment
        # 2. Split that segment by '/' to separate the ID from the filename
        druid_part = url.split("druid:")[-1]
        dataset_id, base_filename = druid_part.split("/", 1)
        
        # Prepend the ID directly to the output file name
        local_filename = f"{dataset_id}__{base_filename}"
    except (ValueError, IndexError):
        # Fallback if the URL structure doesn't match expectations
        local_filename = url.split('/')[-1]

    req = Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    
    try:
        print(f"[STARTING] -> {local_filename}")
        with urlopen(req) as response, open(local_filename, 'wb') as out_file:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out_file.write(chunk)
        print(f"[SUCCESS]  -> {local_filename} saved.")
        return local_filename
    except HTTPError as e:
        print(f"[ERROR]    -> {local_filename} failed with HTTP Status: {e.code}")
    except URLError as e:
        print(f"[ERROR]    -> {local_filename} network failure: {e.reason}")
    except Exception as e:
        print(f"[ERROR]    -> {local_filename} unexpected exception: {e}")
    return None

def main():
    if not os.path.exists(URL_FILE):
        print(f"Error: Target manifest '{URL_FILE}' not found.")
        sys.exit(1)
        
    with open(URL_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
        
    if not urls:
        print(f"Error: '{URL_FILE}' is empty.")
        sys.exit(1)

    print("=========================================")
    print(f" Launching Dynamic-Parsing Downloader")
    print(f" Active Concurrency Pool: {MAX_WORKERS} Threads")
    print(f" Total Files Scheduled:   {len(urls)}")
    print("=========================================\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(download_file, url): url for url in urls if "adaptation" in url}
        
        success_count = 0
        for future in as_completed(future_to_url):
            result = future.result()
            if result:
                success_count += 1

    print("\n=========================================")
    print(f" Execution Cycle Complete.")
    print(f" Successfully saved {success_count}/{len(urls)} assets.")
    print("=========================================")

if __name__ == "__main__":
    main()