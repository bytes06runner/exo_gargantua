"""Take the one-time catalog snapshots listed in docs/sample_definition.md, Section 1.

Small catalogs only (a few MB each); the per-sector MAST product lists and TCE tables are
pulled by the Kaggle freeze job. Each file is saved under data/raw/ with the download date in
its name, and data/raw/snapshots.json records URL, UTC time, SHA-256, size, row count and the
git commit. Refuses to overwrite an existing snapshot (snapshots are taken once).
"""

import datetime as dt
import gzip
import hashlib
import io
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "snapshots.json"
UA = {"User-Agent": "exogargantua-snapshot (mailto:cybrobasics@gmail.com)"}

PS_QUERY = "select * from pscomppars"
SOURCES = {
    "S-TOI": ("exofop_toi", "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv", ".csv"),
    "S-CTOI": ("exofop_ctoi", "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv", ".csv"),
    "S-PS": ("pscomppars", "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
             + urllib.parse.urlencode({"query": PS_QUERY, "format": "csv"}), ".csv"),
    "S-EB": ("tess_ebs_prsa2022_v1.0",
             "https://archive.stsci.edu/hlsps/tess-ebs/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv", ".csv"),
    "S-EB2-new": ("kostov2025_t10k_table3", "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/279/50/table3.dat.gz", ".dat"),
    "S-EB2-known": ("kostov2025_t10k_table4", "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/279/50/table4.dat", ".dat"),
    "S-EB2-readme": ("kostov2025_t10k_ReadMe", "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/279/50/ReadMe", ".txt"),
    "S-G21": ("guerrero2021_toi_table2", "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/254/39/table2.dat.gz", ".dat"),
    "S-G21-readme": ("guerrero2021_toi_ReadMe", "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/254/39/ReadMe", ".txt"),
    "S-LD-ATLAS": ("claret2017_tess_quadratic_atlas_table25", "https://cdsarc.cds.unistra.fr/ftp/J/A+A/600/A30/table25.dat.gz", ".dat"),
    "S-LD-PHOENIX": ("claret2017_tess_quadratic_phoenix_table15", "https://cdsarc.cds.unistra.fr/ftp/J/A+A/600/A30/table15.dat", ".dat"),
    "S-LD-readme": ("claret2017_tess_ReadMe", "https://cdsarc.cds.unistra.fr/ftp/J/A+A/600/A30/ReadMe", ".txt"),
    "S-ORB": ("tess_orbit_times", "https://tess.mit.edu/public/files/TESS_orbit_times.csv", ".csv"),
}


def fetch(url):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                return r.read(), dict(r.headers)
        except Exception:
            if i == 3:
                raise


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    now = dt.datetime.now(dt.timezone.utc)
    day = now.strftime("%Y%m%d")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    for sid, (stem, url, ext) in SOURCES.items():
        if sid in manifest:
            print(f"skip {sid}: already snapshotted as {manifest[sid]['file']}")
            continue
        body, headers = fetch(url)
        if body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        text = body.decode("utf-8", errors="replace")
        if ext == ".csv" and ("<html" in text[:500].lower() or "," not in text.splitlines()[0]):
            raise RuntimeError(f"{sid}: response does not look like CSV")
        out = RAW / f"{stem}_{day}{ext}"
        if len(body) > 20e6:  # keep git-friendly: large snapshots stored gzip-compressed
            out = out.with_name(out.name + ".gz")
            out.write_bytes(gzip.compress(body, mtime=0))
        else:
            out.write_bytes(body)
        n_lines = sum(1 for line in io.StringIO(text) if line.strip())
        manifest[sid] = {
            "file": str(out.relative_to(ROOT)), "url": url, "downloaded_utc": now.isoformat(timespec="seconds"),
            "sha256_uncompressed": hashlib.sha256(body).hexdigest(), "bytes_uncompressed": len(body),
            "data_lines": n_lines - (1 if ext == ".csv" else 0),
            "last_modified": headers.get("Last-Modified"), "git_commit": commit,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=1))  # after each file, so a failure keeps prior entries
        print(f"ok   {sid:12s} {out.name}  {len(body)/1e6:.2f} MB  {manifest[sid]['data_lines']} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
