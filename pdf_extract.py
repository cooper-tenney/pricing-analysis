"""
Extract the first reasonable table from a PDF into a pandas DataFrame.
Tries camelot first (if installed), then pdfplumber.
"""

from pathlib import Path

import pandas as pd


def _looks_like_header(row) -> bool:
    """Heuristic: row looks like column headers if mostly strings and reasonably unique."""
    vals = [str(v).strip() for v in row]
    non_empty = [v for v in vals if v]
    if len(non_empty) < 2:
        return False
    try:
        pd.to_numeric(vals, errors="coerce")
        numeric_count = sum(pd.notna(pd.to_numeric(vals, errors="coerce")))
        return numeric_count < len(vals) * 0.5
    except Exception:
        return True


def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, df.columns != ""]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def extract_table_from_pdf(pdf_path: Path) -> pd.DataFrame:
    """
    Extract the first/largest table from a PDF. Returns DataFrame.
    Raises ValueError with helpful message if extraction fails.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise ValueError(f"PDF not found: {pdf_path}")

    df: pd.DataFrame | None = None

    # Try camelot first
    try:
        import camelot
        tables = camelot.read_pdf(str(pdf_path), pages="1")
        if tables.n > 0:
            largest = max(tables, key=lambda t: t.shape[0] * t.shape[1])
            df = largest.df
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback to pdfplumber
    if df is None or df.empty:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                all_tables = []
                for page in pdf.pages[:2]:
                    for t in page.extract_tables() or []:
                        if t and len(t) >= 2 and len(t[0]):
                            all_tables.append(t)
                if all_tables:
                    rows = max(all_tables, key=lambda r: len(r) * (len(r[0]) if r and r[0] else 0))
                    headers = [str(c or f"col_{i}").strip() for i, c in enumerate(rows[0])]
                    df = pd.DataFrame(rows[1:], columns=headers)
                else:
                    df = pd.DataFrame()
        except ImportError:
            if df is None:
                raise ValueError(
                    "PDF table extraction requires camelot-py[cv] or pdfplumber. "
                    "Install one: pip install pdfplumber (simpler) or pip install 'camelot-py[cv]'. "
                    "Alternatively, export your table to CSV and upload the CSV."
                )
            raise

    if df is None or df.empty:
        raise ValueError(
            "Could not extract a table from this PDF. "
            "Please export your data to CSV and upload the CSV file instead."
        )

    df = _clean_headers(df)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError(
            "Extracted table was empty after cleaning. "
            "Please upload a CSV file instead."
        )
    return df.reset_index(drop=True)
