"""Print arXiv titles, authors, comments and abstracts for a list of IDs.

Usage: python scripts/lit_search/arxiv_abstracts.py 2610.38158 2606.21807 ...
"""
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def main() -> None:
    ids = sys.argv[1:]
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": ",".join(ids), "max_results": len(ids)})
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
        title = " ".join(e.find("a:title", NS).text.split())
        pub, upd = e.find("a:published", NS).text[:10], e.find("a:updated", NS).text[:10]
        authors = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
        au = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        cmt = e.find("arxiv:comment", NS)
        print(f"### {aid} | {pub} (upd {upd}) | {title}\n{au}" + (f" | {cmt.text.strip()}" if cmt is not None else ""))
        print(" ".join(e.find("a:summary", NS).text.split()) + "\n")


if __name__ == "__main__":
    main()
