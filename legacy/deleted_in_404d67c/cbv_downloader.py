import os
import glob
import time
import requests
import astropy.io.fits as fits

CACHE_DIR = os.path.expanduser("~/.lightkurve/cache/mastDownload/TESS")
CBV_DIR = os.path.abspath("./tess_cbv_cache")

def fetch_with_backoff(url, max_retries=10):
    retries = [15, 30, 45, 60, 90, 120, 120, 120, 120, 120]
    for i in range(max_retries):
        try:
            print(f"    Fetching {url}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            if i < max_retries - 1:
                delay = retries[i]
                print(f"    [Error] {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"    [Fatal] Failed to download after {max_retries} retries.")
                return None

def main():
    if not os.path.exists(CBV_DIR):
        os.makedirs(CBV_DIR)
        
    print("Scanning cached light curves to determine required CBVs...")
    files_local = glob.glob(os.path.join("./tess_cache", "**/*lc.fits"), recursive=True)
    files_global = glob.glob(os.path.join(CACHE_DIR, "**/*lc.fits"), recursive=True)
    files = files_local + files_global
    
    required_cbvs = set()
    for f in files:
        try:
            header = fits.getheader(f, 0)
            sector = header.get("SECTOR")
            camera = header.get("CAMERA")
            ccd = header.get("CCD")
            if sector and camera and ccd:
                required_cbvs.add((sector, camera, ccd))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    print(f"Found {len(required_cbvs)} unique (Sector, Camera, CCD) combinations.")
    
    for sector, camera, ccd in sorted(list(required_cbvs)):
        print(f"\nProcessing Sector {sector}, Camera {camera}, CCD {ccd}...")
        
        search_str = f"s{sector:04d}-{camera}-{ccd}-"
        
        # Check if we already have it
        existing = glob.glob(os.path.join(CBV_DIR, f"*{search_str}*.fits"))
        if existing:
            print(f"  Already downloaded: {os.path.basename(existing[0])}")
            continue
            
        script_url = f"https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{sector}_cbv.sh"
        
        # Download the curl script first
        script_content = fetch_with_backoff(script_url, max_retries=5)
        if not script_content:
            print(f"  Failed to get curl script for sector {sector}")
            continue
            
        script_text = script_content.decode('utf-8')
        target_url = None
        target_filename = None
        
        for line in script_text.splitlines():
            if search_str in line and "curl" in line:
                parts = line.split()
                target_url = parts[-1]
                target_filename = parts[-2]
                break
                
        if not target_url:
            print(f"  Could not find {search_str} in the download script.")
            continue
            
        print(f"  Found CBV URL: {target_url}")
        
        fits_content = fetch_with_backoff(target_url, max_retries=10)
        if fits_content:
            out_path = os.path.join(CBV_DIR, target_filename)
            with open(out_path, "wb") as out_f:
                out_f.write(fits_content)
            print(f"  Saved {target_filename} to cache.")

if __name__ == "__main__":
    main()
