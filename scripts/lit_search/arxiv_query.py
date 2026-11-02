"""arXiv API search, newest first (literature passes; docs/new_direction/related_work_paper.md §6).

Usage: python scripts/lit_search/arxiv_query.py SINCE MAX 'query1' 'query2' ...
SINCE is YYYY-MM-DD; only entries submitted on or after it are printed. MAX is the number of
results fetched per query before the date filter. One request per 3 s (arXiv's limit).
A leading * marks an entry not yet printed by an earlier query in the same run.
"""
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}


def main() -> None:
    since, mx, queries = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    seen: set[str] = set()
    for i, q in enumerate(queries):
        if i:
            time.sleep(3.1)
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": q, "sortBy": "submittedDate", "sortOrder": "descending", "max_results": mx})
        data = None
        for attempt in range(3):
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt}: {e}", file=sys.stderr)
                time.sleep(10)
        if data is None:
            print(f"## {q}: FAILED")
            continue
        rows = []
        for e in ET.fromstring(data).findall("a:entry", NS):
            pub = e.find("a:published", NS).text[:10]
            if pub < since:
                continue
            aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
            base = aid.split("v")[0]
            title = " ".join(e.find("a:title", NS).text.split())
            rows.append(f"  {'' if base in seen else '*'}{aid} {pub} {title[:150]}")
            seen.add(base)
        print(f"## {q}  ({len(rows)} since {since})")
        print("\n".join(rows))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
