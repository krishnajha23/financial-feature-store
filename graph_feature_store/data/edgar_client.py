"""
data/edgar_client.py

SEC EDGAR public API client.

Rate limit: 10 req/sec per SEC guidelines.
User-Agent header is required — SEC blocks requests without it.

Usage:
    client = EDGARClient()
    subs   = client.get_company_submissions("0000320193")  # Apple
    facts  = client.get_company_facts("0000320193")
    filings = client.search_filings("insider trading", form_type="4")
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests


class EDGARClient:
    """
    Client for SEC EDGAR public APIs.

    Endpoints:
      /submissions/CIK{cik}.json    - filing history for a company
      /api/xbrl/companyfacts/...    - structured financial data (XBRL)
      EFTS search API               - full-text search across all filings
      RSS feed                      - real-time new filings

    Always set a descriptive User-Agent — SEC requires this and will
    block requests from scripts with no User-Agent or "python-requests/...".
    """

    BASE_URL   = "https://data.sec.gov"
    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    RSS_URL    = "https://www.sec.gov/cgi-bin/browse-edgar"

    def __init__(self, user_agent: str = "Krishna Jha research@gatech.edu"):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent":       user_agent,
            "Accept-Encoding":  "gzip, deflate",
        })
        self._last_request = 0.0

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self):
        """Enforce 10 req/sec per SEC guidelines (100ms minimum spacing)."""
        elapsed = time.time() - self._last_request
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._last_request = time.time()

    # ------------------------------------------------------------------
    # Core API calls
    # ------------------------------------------------------------------

    def get_company_submissions(self, cik: str) -> dict:
        """
        Filing history for a company.
        CIK = Central Index Key, zero-padded to 10 digits.
        Returns all historical filings: form type, accession number, dates.
        """
        self._rate_limit()
        url = f"{self.BASE_URL}/submissions/CIK{cik.zfill(10)}.json"
        r = self.session.get(url)
        r.raise_for_status()
        return r.json()

    def get_company_facts(self, cik: str) -> dict:
        """
        All XBRL financial facts for a company.
        Includes structured data: revenue, employees, executives, filings.
        Used to enrich node features in the graph.
        """
        self._rate_limit()
        url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
        r = self.session.get(url)
        r.raise_for_status()
        return r.json()

    def get_filing_document(self, cik: str, accession_number: str,
                             filename: str) -> str:
        """Download a specific filing document (XML for Form 4, HTML for 10-K)."""
        self._rate_limit()
        acc = accession_number.replace("-", "")
        url = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{acc}/{filename}"
        r = self.session.get(url)
        r.raise_for_status()
        return r.text

    def search_filings(self, query: str, form_type: str = "4",
                        date_from: str = "2020-01-01",
                        date_to:   str = "2024-12-31",
                        hits_per_page: int = 100) -> list[dict]:
        """
        Full-text search across all EDGAR filings.
        Used to find filings mentioning specific companies or executives.
        """
        self._rate_limit()
        params = {
            "q":         f'"{query}"',
            "dateRange": "custom",
            "startdt":   date_from,
            "enddt":     date_to,
            "forms":     form_type,
        }
        r = self.session.get(self.SEARCH_URL, params=params)
        r.raise_for_status()
        return r.json().get("hits", {}).get("hits", [])

    def watch_rss_feed(self, form_types: list[str] = ["4", "DEF 14A", "8-K"]) -> list[dict]:
        """
        Poll EDGAR RSS feed for new filings.
        Called by Kafka producer every ~60s to detect new filings.
        Returns structured list of new filings since last check.
        """
        self._rate_limit()
        params = {
            "action":   "getcurrent",
            "type":     "|".join(form_types),
            "owner":    "include",
            "count":    40,
            "output":   "atom",
        }
        r = self.session.get(self.RSS_URL, params=params)
        r.raise_for_status()
        return self._parse_rss(r.text)

    def _parse_rss(self, rss_xml: str) -> list[dict]:
        root = ET.fromstring(rss_xml)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}
        filings = []
        for entry in root.findall("atom:entry", ns):
            try:
                filings.append({
                    "title":   entry.find("atom:title",   ns).text,
                    "url":     entry.find("atom:link",    ns).get("href"),
                    "updated": entry.find("atom:updated", ns).text,
                    "summary": entry.find("atom:summary", ns).text,
                })
            except AttributeError:
                continue
        return filings

    # ------------------------------------------------------------------
    # Bulk pull for dataset construction
    # ------------------------------------------------------------------

    def pull_company_dataset(self, cik_list: list[str],
                              out_dir: str = "data/raw") -> None:
        """
        Pull submissions + company facts for a list of CIKs.
        Saves to JSON files for offline processing.

        ~1000 companies takes ~15-20 minutes at 10 req/sec rate limit
        (2 requests per company = 2000 requests = ~200s).
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        for i, cik in enumerate(cik_list):
            try:
                subs  = self.get_company_submissions(cik)
                (out / f"{cik}_submissions.json").write_text(
                    json.dumps(subs, indent=2))

                facts = self.get_company_facts(cik)
                (out / f"{cik}_facts.json").write_text(
                    json.dumps(facts, indent=2))

                name = subs.get("name", "Unknown")
                if i % 50 == 0:
                    print(f"[{i}/{len(cik_list)}] {cik}: {name}")

            except requests.HTTPError as e:
                print(f"[warn] CIK {cik}: {e}")
                continue
            except Exception as e:
                print(f"[error] CIK {cik}: {e}")
                continue

        print(f"Pulled {len(cik_list)} companies → {out_dir}/")


# ------------------------------------------------------------------
# S&P 500 CIK list (sample — full list from SEC EDGAR company search)
# ------------------------------------------------------------------

SP500_SAMPLE_CIKS = [
    "0000320193",  # Apple
    "0000789019",  # Microsoft
    "0001018724",  # Amazon
    "0001652044",  # Alphabet
    "0001326801",  # Meta
    "0001045810",  # NVIDIA
    "0000200406",  # J&J
    "0000732717",  # Visa
    "0001403161",  # Berkshire Hathaway (B shares)
    "0000315293",  # JPMorgan Chase
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--companies", type=int, default=50,
                        help="Number of S&P 500 companies to pull")
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()

    client = EDGARClient()
    ciks   = SP500_SAMPLE_CIKS[:args.companies]
    print(f"Pulling {len(ciks)} companies from EDGAR...")
    client.pull_company_dataset(ciks, out_dir=args.out_dir)
