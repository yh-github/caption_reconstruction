"""Semantic Scholar citation chase: papers citing each arXiv ID, newest first.

Usage: python scripts/lit_search/s2_citations.py SINCE ARXIV_ID [ARXIV_ID ...]
Prints citing papers with publicationDate >= SINCE (YYYY-MM-DD), plus those with no date ("????").
Without an API key the endpoint rate-limits (HTTP 429); the script backs off and retries.
"""
import json
import sys
import time
import urllib.request


def get(url: str):
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lit-review"})
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except Exception as e:  # noqa: BLE001
            wait = 5 * (attempt + 1)
            print(f"  [retry {attempt} after {wait}s: {e}]", file=sys.stderr)
            time.sleep(wait)
    return None


def main() -> None:
    since, ids = sys.argv[1], sys.argv[2:]
    for aid in ids:
        rows, offset, failed = [], 0, False
        while True:
            d = get(f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{aid}/citations"
                    f"?fields=title,publicationDate,externalIds,venue&limit=1000&offset={offset}")
            if d is None:
                failed = True
                break
            for c in d.get("data", []):
                p = c["citingPaper"]
                date = p.get("publicationDate") or ""
                if date and date < since:
                    continue
                ax = (p.get("externalIds") or {}).get("ArXiv", "")
                rows.append((date, ax, p.get("venue") or "", p.get("title") or ""))
            if "next" not in d:
                break
            offset = d["next"]
            time.sleep(3)
        if failed:
            print(f"## citing {aid}: FAILED")
            continue
        rows.sort(reverse=True)
        print(f"## citing {aid}: {len(rows)} since {since}")
        for date, ax, venue, title in rows:
            print(f"  {date or '????'} {ax:12} {title[:140]} [{venue[:30]}]")
        sys.stdout.flush()
        time.sleep(4)


if __name__ == "__main__":
    main()
