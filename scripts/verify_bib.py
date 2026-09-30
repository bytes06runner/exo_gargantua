"""Verify that every entry in paper/refs.bib resolves at its registrar.

Checks, per entry:
  * a DOI resolves at Crossref (or DataCite for data DOIs), or an arXiv
    eprint resolves at the arXiv API;
  * the registrar's title matches the BibTeX title (token overlap >= 0.6).
Also checks that every \\cite key used in paper/*.tex exists in refs.bib.

Exit status is non-zero on any failure (CLAUDE.md rule 4).
"""

import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "refs.bib"
UA = {"User-Agent": "exogargantua-bibcheck (mailto:cybrobasics@gmail.com)"}
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def _tokens(s):
    s = re.sub(r"<[^>]+>|\\[a-zA-Z]+|[{}$]", " ", s or "").lower()
    return set(re.findall(r"[a-z0-9]{3,}", s))


def _field(body, name):
    m = re.search(rf"(?:^|[,\s]){name}\s*=\s*\{{", body, flags=re.I)
    if not m:
        return None
    depth, i = 1, m.end()
    while depth and i < len(body):
        depth += {"{": 1, "}": -1}.get(body[i], 0)
        i += 1
    return body[m.end():i - 1].strip()


def parse_bib(text):
    """Balanced-brace BibTeX parser (doi.org returns one-line entries)."""
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        depth, i = 1, m.end()
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[m.end():i - 1]
        entries[m.group(2)] = {f: _field(body, f) for f in ("doi", "eprint", "title")}
    return entries


def registrar_title(entry):
    if entry["doi"]:
        doi = entry["doi"]
        if doi.upper().startswith("10.17909/"):  # MAST DOIs are registered with DataCite
            data = json.loads(_get(f"https://api.datacite.org/dois/{doi}"))
            return data["data"]["attributes"]["titles"][0]["title"]
        data = json.loads(_get(f"https://api.crossref.org/works/{doi}"))
        return (data["message"].get("title") or [""])[0]
    if entry["eprint"]:
        xml = _get(f"http://export.arxiv.org/api/query?id_list={entry['eprint']}")
        e = ET.fromstring(xml).find("a:entry", ATOM)
        return " ".join(e.find("a:title", ATOM).text.split())
    raise ValueError("entry has neither doi nor eprint")


def main():
    entries = parse_bib(BIB.read_text())
    failures = []
    for key, entry in entries.items():
        try:
            title = registrar_title(entry)
            a, b = _tokens(entry["title"]), _tokens(title)
            overlap = len(a & b) / max(1, min(len(a), len(b)))
            status = "ok" if overlap >= 0.6 else "TITLE-MISMATCH"
        except Exception as exc:
            status, title, overlap = f"UNRESOLVED ({exc})", "", 0.0
        if status != "ok":
            failures.append(key)
        print(f"{status:14s} {key:22s} overlap={overlap:.2f} | {title[:80]}")
        time.sleep(0.3)

    cited = set()
    for tex in (ROOT / "paper").glob("*.tex"):
        for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex.read_text()):
            cited.update(k.strip() for k in m.group(1).split(","))
    missing = sorted(cited - set(entries))
    for key in missing:
        print(f"MISSING-IN-BIB  {key}")
    failures += missing

    print(f"\n{len(entries)} entries checked, {len(cited)} cited keys, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
