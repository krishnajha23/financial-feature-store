"""
data/edgar_parser.py

Parsers for SEC EDGAR filing types.

Form 4  — insider trading: reporter, issuer, shares, price, date
DEF 14A — proxy statement: board composition, executive comp, related parties
10-K    — annual report: risk factors, MDA, material events (chunked for RAG)

Output: ParsedFiling dataclass → postgres insert + graph edge extraction.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass
class ParsedFiling:
    accession_number: str
    cik:              str
    form_type:        str
    filed_at:         str
    executives:       list[dict] = field(default_factory=list)
    relationships:    list[dict] = field(default_factory=list)
    trades:           list[dict] = field(default_factory=list)
    text_chunks:      list[dict] = field(default_factory=list)


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------

class EDGARParser:
    """
    Parses EDGAR filings into structured data for:
      - Graph construction (nodes + edges)
      - RAG corpus (text chunks → OpenSearch)
      - PostgreSQL storage (filing metadata)
    """

    CHUNK_SIZE       = 512   # tokens per RAG chunk
    CHUNK_OVERLAP    = 64    # overlap to preserve context across chunks
    MAX_CHUNK_CHARS  = 2000  # approximate character limit per chunk

    def __init__(self, db_conn=None):
        self.db = db_conn

    # ------------------------------------------------------------------
    # Form 4 — insider trading disclosure
    # ------------------------------------------------------------------

    def parse_form4(self, xml_text: str, accession_number: str) -> ParsedFiling:
        """
        Form 4: who traded what in which company.

        Key extracted relationships:
          reporter (executive) -[insider]-> issuer (company)
          → creates TRADED edge in the graph

        XML structure:
          issuerCik, issuerName         — company being traded
          rptOwnerCik, rptOwnerName     — executive who filed
          nonDerivativeTransaction       — stock trade details
        """
        root = ET.fromstring(xml_text)

        issuer_cik    = root.find(".//issuerCik").text.strip()
        issuer_name   = root.find(".//issuerName").text.strip()
        reporter_cik  = root.find(".//rptOwnerCik").text.strip()
        reporter_name = root.find(".//rptOwnerName").text.strip()

        def _find_float(elem, path: str) -> float:
            node = elem.find(path)
            if node is None or not node.text:
                return 0.0
            try:
                return float(node.text.strip())
            except ValueError:
                return 0.0

        trades = []
        for txn in root.findall(".//nonDerivativeTransaction"):
            trade = {
                "accession_number": accession_number,
                "reporter_cik":     reporter_cik,
                "issuer_cik":       issuer_cik,
                "transaction_date": self._find_text(txn, ".//transactionDate/value"),
                "transaction_type": self._find_text(txn, ".//transactionCode"),
                "shares":           _find_float(txn, ".//transactionShares/value"),
                "price_per_share":  _find_float(txn, ".//transactionPricePerShare/value"),
                "shares_after":     _find_float(txn, ".//sharesOwnedFollowingTransaction/value"),
            }
            trades.append(trade)

        relationships = [{
            "reporter_cik": reporter_cik,
            "issuer_cik":   issuer_cik,
            "rel_type":     "insider",
        }]

        return ParsedFiling(
            accession_number=accession_number,
            cik=issuer_cik,
            form_type="4",
            filed_at="",
            executives=[{"name": reporter_name, "cik": reporter_cik}],
            relationships=relationships,
            trades=trades,
        )

    # ------------------------------------------------------------------
    # DEF 14A — proxy statement
    # ------------------------------------------------------------------

    def parse_proxy(self, html_text: str, accession_number: str,
                     cik: str) -> ParsedFiling:
        """
        DEF 14A: board composition, executive compensation, related party transactions.

        Key extracted relationships:
          executive -[board_member]-> company
          executive -[co_director]-> executive  (implicit, two executives at same board)

        HTML parsing: proxy statements are HTML with inconsistent structure.
        We look for the executive compensation and director tables,
        then extract names from them.
        """
        soup = BeautifulSoup(html_text, "html.parser")

        executives    = []
        relationships = []

        # Look for director/executive tables
        # Proxy statements list directors in a table with "Name" column
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 3:
                continue

            header_text = rows[0].get_text().lower() if rows else ""
            if "name" not in header_text and "director" not in header_text:
                continue

            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue
                name = cells[0].get_text(strip=True)
                if not name or len(name) < 3 or name.lower() in ("name", "—", "-"):
                    continue

                # Try to determine title from second cell
                title = cells[1].get_text(strip=True) if len(cells) > 1 else ""

                is_director = any(kw in title.lower()
                                  for kw in ["director", "board", "chair"])

                executives.append({
                    "name":        name,
                    "cik":         cik,
                    "title":       title,
                    "is_director": is_director,
                })
                relationships.append({
                    "executive_name": name,
                    "company_cik":    cik,
                    "rel_type": "board_member" if is_director else "officer",
                })

        # Extract related party transaction text for RAG
        chunks = self._extract_related_party_chunks(soup, accession_number)

        return ParsedFiling(
            accession_number=accession_number,
            cik=cik,
            form_type="DEF 14A",
            filed_at="",
            executives=executives,
            relationships=relationships,
            text_chunks=chunks,
        )

    def _extract_related_party_chunks(self, soup: BeautifulSoup,
                                       accession_number: str) -> list[dict]:
        """Extract related party transaction sections for RAG corpus."""
        chunks = []
        chunk_idx = 0

        # Find sections mentioning related party transactions
        for elem in soup.find_all(["h2", "h3", "h4"]):
            heading = elem.get_text(strip=True).lower()
            if "related party" in heading or "certain relationships" in heading:
                # Grab text from sibling elements until next heading
                text_parts = []
                for sib in elem.find_next_siblings():
                    if sib.name in ["h2", "h3", "h4"]:
                        break
                    text_parts.append(sib.get_text(separator=" ", strip=True))

                full_text = " ".join(text_parts)
                for chunk_text in self._split_into_chunks(full_text):
                    if chunk_text.strip():
                        chunks.append({
                            "accession_number": accession_number,
                            "chunk_index":      chunk_idx,
                            "chunk_text":       chunk_text,
                            "chunk_type":       "related_party",
                        })
                        chunk_idx += 1

        return chunks

    # ------------------------------------------------------------------
    # 10-K — annual report
    # ------------------------------------------------------------------

    def parse_10k(self, html_text: str, accession_number: str,
                   cik: str) -> ParsedFiling:
        """
        10-K: risk factors, MD&A, material events.
        Focus on chunks most relevant to compliance risk.

        Chunking strategy: fixed-size chunks with overlap.
        Chunk type annotation enables targeted retrieval:
        "find risk factor mentions" → filter chunk_type=risk_factor.
        """
        soup   = BeautifulSoup(html_text, "html.parser")
        chunks = []
        idx    = 0

        section_map = {
            "item 1a":    "risk_factor",
            "risk factor": "risk_factor",
            "item 7":     "mda",
            "management": "mda",
            "item 9a":    "internal_controls",
            "item 8":     "financials",
        }

        current_type = "general"

        for elem in soup.find_all(["h1", "h2", "h3", "h4", "p"]):
            text = elem.get_text(separator=" ", strip=True)
            if not text:
                continue

            # Detect section type from heading
            if elem.name in ["h1", "h2", "h3", "h4"]:
                heading_lower = text.lower()
                for kw, chunk_type in section_map.items():
                    if kw in heading_lower:
                        current_type = chunk_type
                        break
                continue

            # Only chunk substantive paragraphs
            if len(text) < 100:
                continue

            for chunk_text in self._split_into_chunks(text):
                chunks.append({
                    "accession_number": accession_number,
                    "chunk_index":      idx,
                    "chunk_text":       chunk_text,
                    "chunk_type":       current_type,
                    "token_count":      len(chunk_text.split()),
                })
                idx += 1

        return ParsedFiling(
            accession_number=accession_number,
            cik=cik,
            form_type="10-K",
            filed_at="",
            text_chunks=chunks,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into overlapping fixed-size chunks by character count."""
        if len(text) <= self.MAX_CHUNK_CHARS:
            return [text]

        chunks = []
        start  = 0
        step   = self.MAX_CHUNK_CHARS - (self.MAX_CHUNK_CHARS // 8)  # ~12.5% overlap

        while start < len(text):
            end  = start + self.MAX_CHUNK_CHARS
            chunk = text[start:end]

            # Trim to last sentence boundary
            last_period = chunk.rfind(". ")
            if last_period > self.MAX_CHUNK_CHARS // 2:
                chunk = chunk[: last_period + 1]

            chunks.append(chunk)
            start += step

        return chunks

    @staticmethod
    def _find_text(elem, path: str, default: str = "") -> str:
        node = elem.find(path)
        if node is None or not node.text:
            return default
        return node.text.strip()

    # ------------------------------------------------------------------
    # PostgreSQL persistence
    # ------------------------------------------------------------------

    def save_to_db(self, filing: ParsedFiling) -> None:
        """Persist parsed filing to PostgreSQL."""
        if not self.db:
            return

        with self.db.cursor() as cur:
            # Upsert company
            for exec_data in filing.executives:
                cik  = exec_data.get("cik", filing.cik)
                name = exec_data.get("name", "")
                if cik:
                    cur.execute("""
                        INSERT INTO companies (cik, name)
                        VALUES (%s, %s)
                        ON CONFLICT (cik) DO NOTHING
                    """, (cik, name))

                # Upsert executive
                if name:
                    cur.execute("""
                        INSERT INTO executives (name, cik, title, is_director)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (name, cik) DO NOTHING
                    """, (name, cik,
                          exec_data.get("title", ""),
                          exec_data.get("is_director", False)))

            # Insert trades
            for trade in filing.trades:
                cur.execute("""
                    INSERT INTO insider_trades
                    (accession_number, reporter_cik, issuer_cik,
                     transaction_date, transaction_type, shares,
                     price_per_share, shares_after)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING
                """, (trade["accession_number"], trade["reporter_cik"],
                      trade["issuer_cik"], trade.get("transaction_date"),
                      trade.get("transaction_type"), trade.get("shares"),
                      trade.get("price_per_share"), trade.get("shares_after")))

            # Insert text chunks
            for chunk in filing.text_chunks:
                cur.execute("""
                    INSERT INTO filing_chunks
                    (accession_number, chunk_index, chunk_text, chunk_type, token_count)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING
                """, (chunk["accession_number"], chunk["chunk_index"],
                      chunk["chunk_text"], chunk.get("chunk_type"),
                      chunk.get("token_count")))

        self.db.commit()
