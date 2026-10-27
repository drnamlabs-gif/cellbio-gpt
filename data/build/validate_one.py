# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Validate-one step: parse ONE PubMed baseline file, then ask PubMed itself
(epost + esearch with the domain MeSH union) which of its PMIDs are in-domain.
Reports yield %, token stats, and sample titles before any bulk download.

Usage: python -I validate_one.py <path-to-pubmed26nXXXX.xml.gz>
"""

import gzip
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNION = (
    '"Morphogenesis"[MeSH] OR "Regeneration"[MeSH] OR '
    '"Cell Physiological Phenomena"[MeSH] OR "Tissue Engineering"[MeSH] OR '
    '"Electrophysiological Phenomena"[MeSH]'
)


def parse_baseline(path):
    docs = {}
    with gzip.open(path, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != "PubmedArticle":
                continue
            pmid = elem.findtext(".//PMID")
            title = elem.findtext(".//ArticleTitle") or ""
            abstract = " ".join(
                (t.text or "") + " ".join(t.itertext())
                for t in elem.findall(".//Abstract/AbstractText")
            ).strip()
            has_mesh = elem.find(".//MeshHeading") is not None
            if pmid:
                docs[pmid] = (title, abstract, has_mesh)
            elem.clear()
    return docs


def post(url, data):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode())
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode()


def matching_pmids(pmids):
    """epost the PMIDs, esearch (UNION) AND #key against them."""
    matched = set()
    for i in range(0, len(pmids), 10000):
        chunk = pmids[i : i + 10000]
        out = post(f"{EUTILS}/epost.fcgi", {"db": "pubmed", "id": ",".join(chunk)})
        qk = re.search(r"<QueryKey>(\d+)</QueryKey>", out).group(1)
        we = re.search(r"<WebEnv>(\S+)</WebEnv>", out).group(1)
        time.sleep(0.4)
        out = post(
            f"{EUTILS}/esearch.fcgi",
            {
                "db": "pubmed",
                "term": f"({UNION}) AND #{qk}",
                "WebEnv": we,
                "usehistory": "y",
                "retmax": 10000,
                "retmode": "json",
            },
        )
        res = json.loads(out)["esearchresult"]
        matched.update(res.get("idlist", []))
        time.sleep(0.4)
    return matched


def main():
    path = sys.argv[1]
    docs = parse_baseline(path)
    with_mesh = [p for p, (_, _, m) in docs.items() if m]
    with_abs = [p for p, (_, a, _) in docs.items() if a]
    print(f"records: {len(docs)}, with MeSH: {len(with_mesh)}, with abstract: {len(with_abs)}")

    matched = matching_pmids(sorted(with_mesh))
    usable = [p for p in matched if docs[p][1]]
    print(f"in-domain (MeSH union): {len(matched)} ({100 * len(matched) / len(docs):.1f}% of file)")
    print(f"in-domain WITH abstract: {len(usable)} ({100 * len(usable) / len(docs):.1f}% of file)")

    toks = [int(len(docs[p][1].split()) * 1.3) for p in usable]
    if toks:
        toks.sort()
        mean = sum(toks) / len(toks)
        print(f"tokens/doc (est): mean {mean:.0f}, p50 {toks[len(toks)//2]}, p90 {toks[int(len(toks)*0.9)]}")
        print(f"projected corpus if yield holds across ~37M PubMed records at this density: see report")
        print("\nsample in-domain titles:")
        for p in usable[:5]:
            print(f"  [{p}] {docs[p][0][:110]}")


if __name__ == "__main__":
    main()
