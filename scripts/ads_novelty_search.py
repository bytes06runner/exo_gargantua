"""ADS full-text novelty search for the Gate 1 check (docs/novelty_check.md, Section 0).

Reads the ADS API token from the ADS_API_TOKEN environment variable (never commit it).
Runs a fixed, committed list of queries, writes the raw responses to
data/cache/lit/ads/ (gitignored) and a de-duplicated hit list to docs/ads_novelty_hits.csv,
marking hits that are already in paper/bib_sources.csv. Screening of the hits is done by a
human and recorded in docs/novelty_check.md.
"""

import csv
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "lit" / "ads"
OUT = ROOT / "docs" / "ads_novelty_hits.csv"
API = "https://api.adsabs.harvard.edu/v1/search/query"
FIELDS = "bibcode,title,first_author,year,doi,identifier,pub"
ROWS = 200

# Fixed before looking at any result. Restricted to refereed papers and arXiv preprints
# in astronomy since TESS launch planning (2014).
BASE = 'collection:astronomy year:2014-2026'
QUERIES = {
    "period_alias_toi_tce": 'full:"period alias" (full:"TOI" OR full:"TCE" OR full:"threshold crossing event")',
    "true_period_half_tess": 'full:("half the true period" OR "half of the true period" OR "twice the true period") full:"TESS"',
    "alias_resolution_transit": 'abs:("alias" OR "aliases" OR "aliasing") abs:("orbital period" OR "true period") abs:("transit" OR "transiting")',
    "alias_stellar_density": 'full:"alias" full:"stellar density" full:"transit" full:"TESS" -full:"duotransit"',
    "alias_probability_calibrated": 'full:("period alias" OR "period aliases") full:("calibrated" OR "calibration") full:("probability" OR "posterior") full:"transit"',
    "harmonic_correction_transit_search": 'full:("harmonic correction" OR "harmonic aliasing" OR "period doubling") full:"transit search" full:"TESS"',
    "toi_period_errors_catalog": 'full:("incorrect period" OR "wrong period" OR "erroneous period") full:("TOI" OR "TESS Objects of Interest")',
    "missing_transits_alias": 'full:("missing transits" OR "empty transits" OR "missing transit") full:"alias" full:"TESS"',
}


def query(token, q):
    """All pages of one query (ADS caps rows per request)."""
    docs, start = [], 0
    while True:
        params = urllib.parse.urlencode({"q": f"{q} {BASE}", "fl": FIELDS, "rows": ROWS,
                                         "start": start, "sort": "date desc"})
        req = urllib.request.Request(f"{API}?{params}", headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        docs += data["response"]["docs"]
        start += ROWS
        if start >= data["response"]["numFound"]:
            data["response"]["docs"] = docs
            return data
        time.sleep(1)


def main():
    token = os.environ.get("ADS_API_TOKEN")
    if not token:
        print("ADS_API_TOKEN is not set; export it in your shell (never commit it).")
        return 2
    known = set()
    with open(ROOT / "paper" / "bib_sources.csv") as fh:
        for row in csv.DictReader(fh):
            known.add(row["id"].lower())

    CACHE.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    hits = {}
    for name, q in QUERIES.items():
        data = query(token, q)
        (CACHE / f"{stamp}_{name}.json").write_text(json.dumps(data, indent=1))
        docs = data["response"]["docs"]
        print(f"{name:38s} numFound={data['response']['numFound']:5d} returned={len(docs)}")
        for d in docs:
            h = hits.setdefault(d["bibcode"], {
                "bibcode": d["bibcode"], "year": d.get("year", ""), "first_author": d.get("first_author", ""),
                "title": (d.get("title") or [""])[0], "pub": d.get("pub", ""),
                "doi": (d.get("doi") or [""])[0], "queries": [],
            })
            h["queries"].append(name)
        time.sleep(1)

    for h in hits.values():
        h["in_bib"] = "y" if (h["doi"] and h["doi"].lower() in known) else "n"
        h["n_queries"] = len(h["queries"])
        h["queries"] = ";".join(h["queries"])

    # most-matched first, then newest
    rows = sorted(hits.values(), key=lambda h: (-h["n_queries"], -int(h["year"] or 0)))
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bibcode", "year", "first_author", "title", "pub", "doi",
                                           "in_bib", "n_queries", "queries", "screened", "relevance_note"])
        w.writeheader()
        for h in rows:
            w.writerow(dict(h, screened="", relevance_note=""))
    print(f"\n{len(rows)} unique hits -> {OUT.relative_to(ROOT)} (search run {stamp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
