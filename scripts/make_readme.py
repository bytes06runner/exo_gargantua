"""Build README.md from docs/README.template.md. Every number in the README is filled in from paper/generated_macros.tex,
the same file the manuscript uses, so the README cannot drift from the results (rule 1 of the repository).

Placeholders:  {{name}}       the macro value as plain text (e.g. 1,341 or 0.786)
               {{pct:name}}   a fraction macro shown as a percentage with one decimal (0.786 -> 78.6%)
               {{p:name}}     a p-value macro shown as " = 5 × 10⁻²⁴" or " = 0.064"
Figures in docs/figures/*.png are rendered from paper/figures/*.pdf with macOS `sips` when it is available.
Usage: python scripts/make_readme.py
"""

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def macros():
    text = (ROOT / "paper/generated_macros.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", text))


def plain(v):
    v = v.replace("{,}", ",").replace("$-$", "-").replace("\\%", "%").replace("\\,", " ").replace("~", " ")
    v = re.sub(r"\\times10\^\{(-?\d+)\}", lambda m: " × 10" + m.group(1).translate(SUP), v)
    v = re.sub(r"10\^\{(-?\d+)\}", lambda m: "10" + m.group(1).translate(SUP), v)
    return v.replace("$", "")


def pvalue(v):
    v = plain(v).strip()
    if v.startswith("="):
        return " = " + v[1:].strip()
    if v.startswith("<"):
        return " < " + v[1:].strip()
    return " = " + v


def fill(template, m):
    def sub(match):
        kind, name = (match.group(1) or ""), match.group(2)
        if name not in m:
            raise SystemExit(f"README placeholder {name!r} has no macro in paper/generated_macros.tex")
        v = m[name]
        if kind == "pct:":
            return f"{100 * float(plain(v)):.1f}%"
        if kind == "p:":
            return pvalue(v)
        return plain(v)
    return re.sub(r"\{\{(pct:|p:)?([A-Za-z]+)\}\}", sub, template)


def figures():
    if not shutil.which("sips"):
        return
    out = ROOT / "docs/figures"
    out.mkdir(exist_ok=True)
    for pdf in sorted((ROOT / "paper/figures").glob("*.pdf")):
        subprocess.run(["sips", "-s", "format", "png", str(pdf), "--out", str(out / (pdf.stem + ".png")), "-Z", "1600"],
                       check=True, capture_output=True)


def main():
    figures()
    text = fill((ROOT / "docs/README.template.md").read_text(), macros())
    left = re.findall(r"\{\{[^}]*\}\}", text)
    assert not left, left
    (ROOT / "README.md").write_text(text)
    print(f"README.md written ({len(text.split())} words)")


if __name__ == "__main__":
    main()
