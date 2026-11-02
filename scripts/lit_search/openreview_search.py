"""OpenReview full-text search over one venue's submissions (default ICLR 2027).

Usage:
  python scripts/lit_search/openreview_search.py search N 'term 1' 'term 2' ...   # id + title; * = new in this run
  python scripts/lit_search/openreview_search.py abs 'exact title 1' ...           # title, keywords, abstract
Set OR_GROUP to search another venue, e.g. OR_GROUP=ICLR.cc/2028/Conference.

notes?id= needs a login (HTTP 403), but the search endpoint returns each note's full content,
so 'abs' looks a paper up by its title. PDFs (openreview.net/pdf?id=...) also return 403 to scripts.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

GROUP = os.environ.get("OR_GROUP", "ICLR.cc/2027/Conference")


def get(url: str) -> dict:
    for attempt in range(4):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=60).read())
        except Exception as e:  # noqa: BLE001
            print(f"  [retry {attempt}: {e}]", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return {}


def val(content: dict, key: str):
    v = content.get(key, "")
    return v.get("value", "") if isinstance(v, dict) else v


def search(term: str, limit: int) -> dict:
    return get("https://api2.openreview.net/notes/search?" + urllib.parse.urlencode(
        {"term": term, "group": GROUP, "limit": limit, "type": "terms"}))


def main() -> None:
    mode = sys.argv[1]
    if mode == "search":
        n, terms = int(sys.argv[2]), sys.argv[3:]
        seen: set[str] = set()
        for t in terms:
            d = search(t, n)
            print(f"## {t} (count {d.get('count')})")
            for note in d.get("notes", []):
                nid = note["id"]
                print(f"  {'' if nid in seen else '*'}{nid} {str(val(note.get('content', {}), 'title'))[:150]}")
                seen.add(nid)
            sys.stdout.flush()
            time.sleep(1.5)
    elif mode == "abs":
        for title in sys.argv[2:]:
            for note in search(title, 3).get("notes", [])[:1]:
                c = note.get("content", {})
                print(f"### {note['id']} | {val(c, 'title')}\nkeywords: {val(c, 'keywords')}")
                print(" ".join(str(val(c, "abstract")).split()) + "\n")
            sys.stdout.flush()
            time.sleep(1.5)
    else:
        sys.exit(f"unknown mode {mode!r}; use 'search' or 'abs'")


if __name__ == "__main__":
    main()
