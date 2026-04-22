-- schema.sql
-- SEC EDGAR filing metadata for graph construction and RAG
-- Run: psql -U postgres -d edgar -f schema.sql

CREATE TABLE IF NOT EXISTS companies (
    cik             VARCHAR(10) PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    ticker          VARCHAR(10),
    sic_code        VARCHAR(4),
    sic_description VARCHAR(255),
    state           VARCHAR(2),
    fiscal_year_end VARCHAR(4),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker);
CREATE INDEX IF NOT EXISTS idx_companies_sic    ON companies(sic_code);

CREATE TABLE IF NOT EXISTS executives (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    cik         VARCHAR(10) REFERENCES companies(cik),
    title       VARCHAR(255),
    is_director BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE(name, cik)
);

CREATE INDEX IF NOT EXISTS idx_executives_name ON executives(name);
CREATE INDEX IF NOT EXISTS idx_executives_cik  ON executives(cik);

CREATE TABLE IF NOT EXISTS filings (
    accession_number VARCHAR(20) PRIMARY KEY,
    cik              VARCHAR(10) REFERENCES companies(cik),
    form_type        VARCHAR(20) NOT NULL,
    filed_at         TIMESTAMP NOT NULL,
    period_of_report DATE,
    document_url     TEXT,
    processed        BOOLEAN DEFAULT FALSE,
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_filings_cik       ON filings(cik);
CREATE INDEX IF NOT EXISTS idx_filings_form_type ON filings(form_type);
CREATE INDEX IF NOT EXISTS idx_filings_filed_at  ON filings(filed_at);
CREATE INDEX IF NOT EXISTS idx_filings_processed ON filings(processed);

-- Form 4 insider trading disclosures
CREATE TABLE IF NOT EXISTS insider_trades (
    id               SERIAL PRIMARY KEY,
    accession_number VARCHAR(20) REFERENCES filings(accession_number),
    reporter_cik     VARCHAR(10),
    issuer_cik       VARCHAR(10) REFERENCES companies(cik),
    transaction_date DATE,
    transaction_type VARCHAR(10),   -- P=purchase, S=sale
    shares           NUMERIC(15, 2),
    price_per_share  NUMERIC(10, 4),
    shares_after     NUMERIC(15, 2),
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_issuer_cik ON insider_trades(issuer_cik);
CREATE INDEX IF NOT EXISTS idx_trades_reporter   ON insider_trades(reporter_cik);
CREATE INDEX IF NOT EXISTS idx_trades_date       ON insider_trades(transaction_date);

-- Board and executive relationships
CREATE TABLE IF NOT EXISTS relationships (
    id            SERIAL PRIMARY KEY,
    executive_id  INTEGER REFERENCES executives(id),
    company_cik   VARCHAR(10) REFERENCES companies(cik),
    rel_type      VARCHAR(50),   -- board_member, officer, director, insider
    start_date    DATE,
    end_date      DATE,
    source_filing VARCHAR(20) REFERENCES filings(accession_number),
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_relationships_executive ON relationships(executive_id);
CREATE INDEX IF NOT EXISTS idx_relationships_company   ON relationships(company_cik);

-- Filing text chunks for RAG corpus
CREATE TABLE IF NOT EXISTS filing_chunks (
    id               SERIAL PRIMARY KEY,
    accession_number VARCHAR(20) REFERENCES filings(accession_number),
    chunk_index      INTEGER NOT NULL,
    chunk_text       TEXT NOT NULL,
    chunk_type       VARCHAR(50),   -- risk_factor, mda, executive_comp, related_party
    token_count      INTEGER,
    embedding_id     VARCHAR(100),  -- OpenSearch document ID
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_accession ON filing_chunks(accession_number);
CREATE INDEX IF NOT EXISTS idx_chunks_type      ON filing_chunks(chunk_type);
