# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Bulk-fetch the cell/tissue-biology pretraining corpus from PubMed.

NCBI caps PubMed efetch at the first 10,000 records of any result set
(retstart >= 10000 returns HTTP 400). So: bisect the query by Entrez date
(EDAT) until each slice holds < 10k records, then efetch each slice whole.

Usage: python -I fetch_corpus.py <out_dir>
Writes: <out_dir>/slice_<from>_<to>.jsonl.gz, one {"pmid","title","abstract"}
per line. Resumable: finished slices are skipped by filename.
"""

import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNION = (
    '("Morphogenesis"[MeSH] OR "Regeneration"[MeSH] OR '
    '"Cell Physiological Phenomena"[MeSH] OR "Tissue Engineering"[MeSH] OR '
    '"Electrophysiological Phenomena"[MeSH]) AND hasabstract'
)
CAP = 9999
SLEEP = 0.55  # stay safely under NCBI's 3 req/s no-key limit
DATE0 = dt.date(1800, 1, 1)
DATE1 = dt.date(2026, 12, 31)


def call(endpoint, data, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{EUTILS}/{endpoint}", data=urllib.parse.urlencode(data).encode()
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  retry {attempt + 1} after {e!r}, sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"gave up on {endpoint}")


def esearch(d0, d1, history=False):
    data = {
        "db": "pubmed", "term": UNION, "datetype": "edat",
        "mindate": d0.strftime("%Y/%m/%d"), "maxdate": d1.strftime("%Y/%m/%d"),
        "retmax": 0, "retmode": "json",
    }
    if history:
        data["usehistory"] = "y"
    # rate-limit responses are valid JSON without "esearchresult"/"count": retry those too
    for attempt in range(6):
        res = json.loads(call("esearch.fcgi", data).decode()).get("esearchresult", {})
        time.sleep(SLEEP)
        if "count" in res:
            if history:
                return int(res["count"]), res.get("querykey"), res.get("webenv")
            return int(res["count"])
        print(f"  esearch no-count response, retry {attempt + 1}", flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"esearch {d0}..{d1}: no count after retries")


def fetch_slice(d0, d1, count, out_dir):
    path = os.path.join(out_dir, f"slice_{d0:%Y%m%d}_{d1:%Y%m%d}.jsonl.gz")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return 0
    _, qk, we = esearch(d0, d1, history=True)
    # the server sometimes truncates a 200 response mid-XML; retry fetch+parse together
    for attempt in range(5):
        xml_bytes = call("efetch.fcgi", {
            "db": "pubmed", "query_key": qk, "WebEnv": we,
            "retstart": 0, "retmax": min(count, 10000), "retmode": "xml",
        })
        time.sleep(SLEEP)
        try:
            root = ET.fromstring(xml_bytes)
            break
        except ET.ParseError as e:
            print(f"  truncated XML ({e}), retry {attempt + 1}", flush=True)
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"slice {d0}..{d1}: unparseable after retries")
    n = 0
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as out_f:
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = art.findtext(".//ArticleTitle") or ""
            abstract = " ".join(
                " ".join(t.itertext()) for t in art.findall(".//Abstract/AbstractText")
            ).strip()
            if pmid and abstract:
                out_f.write(json.dumps(
                    {"pmid": pmid, "title": title, "abstract": abstract},
                    ensure_ascii=False) + "\n")
                n += 1
    os.replace(tmp, path)
    return n


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    total = esearch(DATE0, DATE1)
    print(f"total in-domain records: {total}", flush=True)

    stack = [(DATE0, DATE1, total)]
    done_docs = done_slices = 0
    while stack:
        d0, d1, count = stack.pop()
        if count == 0:
            continue
        if count > CAP and d0 < d1:
            mid = d0 + (d1 - d0) / 2
            left = esearch(d0, mid)
            right = esearch(mid + dt.timedelta(days=1), d1)
            if left + right != count:
                print(f"  note: {d0}..{d1} count {count} != {left}+{right} (EDAT quirks, proceeding)", flush=True)
            stack.append((mid + dt.timedelta(days=1), d1, right))
            stack.append((d0, mid, left))
            continue
        if count > CAP:
            print(f"  WARNING: single day {d0} has {count} > {CAP}; fetching first 10k only", flush=True)
        try:
            n = fetch_slice(d0, d1, count, out_dir)
        except Exception as e:
            # never let one bad slice kill the run: log it, move on, re-run picks it up
            print(f"  SLICE FAILED {d0}..{d1}: {e!r} — logged, continuing", flush=True)
            with open(os.path.join(out_dir, "failed_slices.txt"), "a") as fl:
                fl.write(f"{d0} {d1} {count} {e!r}\n")
            continue
        done_slices += 1
        done_docs += n
        print(f"slice {d0}..{d1}: {n} docs (slices {done_slices}, docs {done_docs})", flush=True)

    print(f"done. {done_docs} docs this run.", flush=True)


if __name__ == "__main__":
    main()
