"""Download recent arXiv abstracts and save them to data/papers.jsonl.

Usage:  python scripts/fetch_papers.py --category cs.CL --n 200

The arXiv API returns XML in the "Atom" feed format. We parse it with Python's built-in
xml library. arXiv asks users to make at most one request every 3 seconds.
"""

import argparse
import xml.etree.ElementTree as ET

import requests

from rag_lab.config import PAPERS_PATH
from rag_lab.data import save_jsonl

API_URL = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def clean(text: str) -> str:
    return " ".join(text.split())  # collapse newlines and double spaces


def parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", NS):
        # <id>http://arxiv.org/abs/2511.01234v1</id>  ->  "2511.01234"
        raw_id = entry.findtext("atom:id", default="", namespaces=NS)
        paper_id = raw_id.rsplit("/abs/", 1)[-1].rsplit("v", 1)[0]
        category = entry.find("arxiv:primary_category", NS)
        papers.append(
            {
                "id": paper_id,
                "title": clean(entry.findtext("atom:title", default="", namespaces=NS)),
                "abstract": clean(entry.findtext("atom:summary", default="", namespaces=NS)),
                "authors": [
                    clean(a.findtext("atom:name", default="", namespaces=NS))
                    for a in entry.findall("atom:author", NS)
                ],
                "published": entry.findtext("atom:published", default="", namespaces=NS)[:10],
                "category": category.get("term") if category is not None else "",
            }
        )
    return papers


def fetch(category: str, n: int) -> list[dict]:
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": n,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(
        API_URL,
        params=params,
        timeout=60,
        headers={"User-Agent": "rag-eval-lab (learning project)"},
    )
    response.raise_for_status()  # turn HTTP errors (4xx/5xx) into a Python exception
    return parse_feed(response.text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="cs.CL")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    papers = fetch(args.category, args.n)
    papers = [p for p in papers if p["id"] and p["abstract"]]  # drop broken entries
    save_jsonl(papers, PAPERS_PATH)
    print(f"Saved {len(papers)} papers to {PAPERS_PATH}")


if __name__ == "__main__":
    main()