"""
Dataset Validation Agent
========================
Validates a CSV dataset before data analysis across 9 check categories.
Opens an HTML report in the browser showing all issues found and how to fix them.

Usage:
    python dataset_validator.py path/to/file.csv
    python dataset_validator.py data.csv --columns order_id customer_id price quantity
    python dataset_validator.py data.csv --output report.html
"""

import sys
import re
import argparse
import webbrowser
import tempfile
import os
from pathlib import Path
from datetime import datetime
from html import escape

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Install dependencies: pip install pandas numpy")
    sys.exit(1)


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

CRITICAL_FIELDS_KEYWORDS = [
    "date", "time", "amount", "total", "revenue",
    "price", "quantity", "qty", "customer", "client",
    "product", "order", "category", "region", "id",
]

DATE_KEYWORDS     = ["date", "time", "period", "month", "year", "timestamp", "created", "updated"]
ID_KEYWORDS       = ["_id", "id", "identifier", "code", "num", "number", "key"]
NUMERIC_KEYWORDS  = ["amount", "price", "qty", "quantity", "total", "sum", "count",
                     "revenue", "cost", "discount", "rate", "percent", "value", "sales", "profit"]
CATEGORY_KEYWORDS = ["category", "type", "status", "method", "region", "country",
                     "city", "segment", "channel", "gender", "department", "payment"]

PSEUDO_NULL_VALUES = {"n/a", "na", "nan", "null", "none", "nil", "-", "--", "---",
                      "unknown", "missing", "empty", "", " ", "not available", "undefined"}

TECHNICAL_COLUMN_PATTERNS = [
    r"^unnamed",
    r"^index$",
    r"^\.index$",
    r"^col\d+$",
    r"^field\d+$",
    r"^column\d+$",
    r"^_c\d+$",
]

SEVERITY_ORDER = {"CRITICAL": 0, "WARNING": 1, "INFO": 2, "OK": 3}


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _matches_any(name: str, keywords: list) -> bool:
    n = name.lower()
    return any(kw in n for kw in keywords)


def _is_id_column(name: str) -> bool:
    return _matches_any(name, ID_KEYWORDS)


def _is_numeric_keyword(name: str) -> bool:
    return _matches_any(name, NUMERIC_KEYWORDS)


def _is_category_column(name: str, series: pd.Series) -> bool:
    if _matches_any(name, CATEGORY_KEYWORDS):
        return True
    dtype_str = str(series.dtype)
    if dtype_str == "object" or dtype_str == "string":
        nunique = series.nunique()
        n = len(series.dropna())
        if n > 0 and nunique / n < 0.1 and nunique <= 50:
            return True
    return False


def _is_critical_field(name: str) -> bool:
    return _matches_any(name, CRITICAL_FIELDS_KEYWORDS)


def _pct(val, total) -> str:
    if total == 0:
        return "0.0%"
    return f"{val / total * 100:.1f}%"


def _is_text_dtype(series: pd.Series) -> bool:
    """Returns True for object or pandas StringDtype columns."""
    return series.dtype == object or str(series.dtype) == "string"


def _try_parse_dates(series: pd.Series):
    """Returns (parsed_series, success_ratio, detected_format)."""
    common_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d.%m.%Y", "%Y/%m/%d", "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
    ]
    sample = series.dropna().astype(str).head(200)
    best_parsed, best_ratio, best_fmt = None, 0.0, None

    for fmt in common_formats:
        try:
            parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
            ratio = parsed.notna().mean()
            if ratio > best_ratio:
                best_ratio, best_parsed, best_fmt = ratio, parsed, fmt
        except Exception:
            continue

    # Fallback: let pandas infer
    try:
        parsed_inf = pd.to_datetime(sample, errors="coerce")
        ratio_inf = parsed_inf.notna().mean()
        if ratio_inf > best_ratio:
            best_ratio, best_parsed, best_fmt = ratio_inf, parsed_inf, "auto-inferred"
    except Exception:
        pass

    return best_parsed, best_ratio, best_fmt


def _safe_date_str(ts) -> str:
    """Convert a Timestamp or any value to a YYYY-MM-DD string safely."""
    try:
        return str(pd.Timestamp(ts).date())
    except Exception:
        return str(ts)


def _safe_days_between(ts_max, ts_min) -> str:
    try:
        return str((pd.Timestamp(ts_max) - pd.Timestamp(ts_min)).days)
    except Exception:
        return "?"


# ─────────────────────────────────────────────
#  Report model
# ─────────────────────────────────────────────

class ValidationReport:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.sections: list = []
        self.summary_counts = {"CRITICAL": 0, "WARNING": 0, "INFO": 0, "OK": 0}

    def add_issue(self, section: str, severity: str, message: str,
                  fix: str = "", details: str = ""):
        self.sections.append({
            "section": section,
            "severity": severity,
            "message": message,
            "fix": fix,
            "details": details,
        })
        self.summary_counts[severity] = self.summary_counts.get(severity, 0) + 1

    def render_html(self) -> str:
        c = self.summary_counts
        total_issues = c.get("CRITICAL", 0) + c.get("WARNING", 0)

        if total_issues == 0:
            status_msg, status_class, status_icon = (
                "Dataset passed validation with no serious issues.", "status-ok", "&#10003;")
        elif c.get("CRITICAL", 0) > 0:
            status_msg, status_class, status_icon = (
                "CRITICAL issues found. Analysis without fixes may produce incorrect results.",
                "status-critical", "&#9888;")
        else:
            status_msg, status_class, status_icon = (
                "Warnings found. It is recommended to fix them before analysis.",
                "status-warning", "&#9888;")

        # Group by section
        from collections import OrderedDict
        by_section = OrderedDict()
        for item in self.sections:
            by_section.setdefault(item["section"], []).append(item)

        sections_html = ""
        for sec_name, items in by_section.items():
            items_html = ""
            sorted_items = sorted(items, key=lambda x: SEVERITY_ORDER.get(x["severity"], 9))
            for item in sorted_items:
                sev = item["severity"]
                badge_map = {
                    "CRITICAL": '<span class="badge badge-critical">CRITICAL</span>',
                    "WARNING":  '<span class="badge badge-warning">WARNING</span>',
                    "INFO":     '<span class="badge badge-info">INFO</span>',
                    "OK":       '<span class="badge badge-ok">OK</span>',
                }
                badge = badge_map.get(sev, "")
                details_block = (
                    f'<pre class="details-block">{escape(item["details"])}</pre>'
                    if item["details"] else ""
                )
                fix_block = (
                    f'<div class="fix-block"><span class="fix-label">&#128161; HOW TO FIX:</span> {escape(item["fix"])}</div>'
                    if item["fix"] else ""
                )
                items_html += f"""
                <div class="card card-{sev.lower()}">
                    <div class="card-header">{badge}<span class="card-message">{escape(item["message"])}</span></div>
                    {details_block}
                    {fix_block}
                </div>"""

            sections_html += f"""
            <div class="section">
                <h2 class="section-title">{escape(sec_name)}</h2>
                <div class="section-body">{items_html}</div>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Validation Report</title>
    <style>
        :root {{
            --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a;
            --border: #2e3248; --text: #e2e8f0; --text-muted: #8892aa;
            --critical: #ff4d6d; --critical-bg: #2a1020; --critical-bd: #5c1a2e;
            --warning:  #f59e0b; --warning-bg:  #1f1a09; --warning-bd:  #4a3a00;
            --info:     #60a5fa; --info-bg:     #0d1a2e; --info-bd:     #1e3a5c;
            --ok:       #34d399; --ok-bg:       #071f16; --ok-bd:       #0e4030;
            --mono: 'JetBrains Mono','Fira Code','Cascadia Code',monospace;
            --sans: 'Inter','Segoe UI',system-ui,sans-serif;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg,#1a1d27 0%,#12152a 100%); border-bottom: 1px solid var(--border); padding: 32px 48px 28px; }}
        .header-top {{ display: flex; align-items: flex-start; gap: 16px; }}
        .header-icon {{ font-size: 40px; line-height: 1; }}
        .header-text h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -.3px; color: #fff; margin-bottom: 4px; }}
        .header-meta {{ color: var(--text-muted); font-size: 13px; }}
        .header-meta span {{ margin-right: 20px; }}
        .header-meta code {{ font-family: var(--mono); font-size: 12px; background: #2e3248; padding: 1px 6px; border-radius: 4px; color: #a8b4d0; }}
        .status-bar {{ margin: 24px 48px 0; padding: 14px 20px; border-radius: 8px; font-size: 14px; font-weight: 500; display: flex; align-items: center; gap: 10px; }}
        .status-critical {{ background: var(--critical-bg); border: 1px solid var(--critical-bd); color: var(--critical); }}
        .status-warning  {{ background: var(--warning-bg);  border: 1px solid var(--warning-bd);  color: var(--warning); }}
        .status-ok       {{ background: var(--ok-bg);       border: 1px solid var(--ok-bd);       color: var(--ok); }}
        .summary {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; padding: 24px 48px; border-bottom: 1px solid var(--border); }}
        .summary-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; display: flex; flex-direction: column; gap: 4px; }}
        .summary-card .count {{ font-size: 32px; font-weight: 700; line-height: 1; font-family: var(--mono); }}
        .summary-card .label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .8px; }}
        .sc-critical .count {{ color: var(--critical); }}
        .sc-warning  .count {{ color: var(--warning); }}
        .sc-info     .count {{ color: var(--info); }}
        .sc-ok       .count {{ color: var(--ok); }}
        .content {{ padding: 24px 48px 48px; }}
        .section {{ margin-bottom: 32px; }}
        .section-title {{ font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 12px; }}
        .card {{ border-radius: 8px; padding: 14px 16px; margin-bottom: 8px; border: 1px solid transparent; border-left-width: 3px; }}
        .card-critical {{ background: var(--critical-bg); border-color: var(--critical-bd); border-left-color: var(--critical); }}
        .card-warning  {{ background: var(--warning-bg);  border-color: var(--warning-bd);  border-left-color: var(--warning); }}
        .card-info     {{ background: var(--info-bg);     border-color: var(--info-bd);     border-left-color: var(--info); }}
        .card-ok       {{ background: var(--ok-bg);       border-color: var(--ok-bd);       border-left-color: var(--ok); }}
        .card-header   {{ display: flex; align-items: flex-start; gap: 10px; }}
        .card-message  {{ flex: 1; color: var(--text); line-height: 1.5; }}
        .badge {{ font-size: 10px; font-weight: 700; letter-spacing: .8px; padding: 2px 8px; border-radius: 4px; text-transform: uppercase; white-space: nowrap; flex-shrink: 0; margin-top: 1px; }}
        .badge-critical {{ background: var(--critical); color: #0f0005; }}
        .badge-warning  {{ background: var(--warning);  color: #0f0a00; }}
        .badge-info     {{ background: var(--info);     color: #000d1f; }}
        .badge-ok       {{ background: var(--ok);       color: #001a0e; }}
        .details-block {{ margin-top: 10px; background: rgba(0,0,0,.25); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; font-family: var(--mono); font-size: 12px; color: #a8b4d0; overflow-x: auto; white-space: pre-wrap; }}
        .fix-block {{ margin-top: 10px; background: rgba(0,0,0,.2); border: 1px solid #2a3060; border-radius: 6px; padding: 9px 14px; font-size: 13px; color: #c8d8f0; }}
        .fix-label {{ font-weight: 600; color: #7eb8ff; }}
        @media (max-width: 768px) {{
            .header, .summary, .content {{ padding-left: 20px; padding-right: 20px; }}
            .status-bar {{ margin-left: 20px; margin-right: 20px; }}
            .summary {{ grid-template-columns: repeat(2,1fr); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-top">
            <div class="header-icon">&#128269;</div>
            <div class="header-text">
                <h1>Dataset Validation Report</h1>
                <div class="header-meta">
                    <span>File: <code>{escape(self.filepath)}</code></span>
                    <span>Generated: {self.timestamp}</span>
                </div>
            </div>
        </div>
        <div class="status-bar {status_class}">
            <span>{status_icon}</span>
            <span>{escape(status_msg)}</span>
        </div>
    </div>
    <div class="summary">
        <div class="summary-card sc-critical"><span class="count">{c.get('CRITICAL',0)}</span><span class="label">Critical</span></div>
        <div class="summary-card sc-warning"> <span class="count">{c.get('WARNING',0)}</span> <span class="label">Warnings</span></div>
        <div class="summary-card sc-info">    <span class="count">{c.get('INFO',0)}</span>    <span class="label">Info</span></div>
        <div class="summary-card sc-ok">      <span class="count">{c.get('OK',0)}</span>      <span class="label">Passed</span></div>
    </div>
    <div class="content">{sections_html}</div>
</body>
</html>"""


# ─────────────────────────────────────────────
#  Validation Agent
# ─────────────────────────────────────────────

class DatasetValidationAgent:

    def __init__(self, filepath: str, expected_columns: list = None):
        self.filepath = filepath
        self.expected_columns = expected_columns or []
        self.df = None
        self.report = ValidationReport(filepath)

    # ── Load ─────────────────────────────────────────────────────────────────

    def _load(self) -> bool:
        encodings  = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]
        separators = [",", ";", "\t", "|"]
        path = Path(self.filepath)

        if not path.exists():
            self.report.add_issue("1. DATASET STRUCTURE", "CRITICAL",
                                  f"File not found: {self.filepath}",
                                  fix="Check the file path.")
            return False

        if path.suffix.lower() != ".csv":
            self.report.add_issue("1. DATASET STRUCTURE", "WARNING",
                                  f"File extension is not .csv: {path.suffix}",
                                  fix="Make sure the file is a valid CSV.")

        for enc in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(self.filepath, encoding=enc, sep=sep, low_memory=False)
                    if df.shape[1] > 1 or sep == ",":
                        self.df = df
                        self.report.add_issue(
                            "1. DATASET STRUCTURE", "INFO",
                            f"File loaded. Encoding: {enc}, separator: '{sep}'"
                        )
                        return True
                except Exception:
                    continue

        self.report.add_issue(
            "1. DATASET STRUCTURE", "CRITICAL",
            "Could not load the file with any standard encoding or separator.",
            fix="Check the file encoding (UTF-8 recommended) and separator format."
        )
        return False

    # ── 1. Structure ─────────────────────────────────────────────────────────

    def check_structure(self):
        df = self.df
        section = "1. DATASET STRUCTURE"

        self.report.add_issue(section, "INFO",
                              f"Dataset size: {df.shape[0]:,} rows x {df.shape[1]} columns")

        if df.empty:
            self.report.add_issue(section, "CRITICAL", "Dataset is empty (0 rows).",
                                  fix="Check the data source.")
            return

        if df.shape[0] < 5:
            self.report.add_issue(section, "WARNING",
                                  f"Very few rows: {df.shape[0]}. Analysis may be unreliable.")

        unnamed = [c for c in df.columns if str(c).strip() == ""]
        if unnamed:
            self.report.add_issue(section, "CRITICAL",
                                  f"Columns with no name found: {len(unnamed)}",
                                  fix="Add headers or remove extra columns from the CSV.")

        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            self.report.add_issue(section, "WARNING",
                                  f"Fully empty columns: {empty_cols}",
                                  fix=f"Remove them: df.drop(columns={empty_cols}, inplace=True)")

        dupes = list({c for c in df.columns if list(df.columns).count(c) > 1})
        if dupes:
            self.report.add_issue(section, "CRITICAL",
                                  f"Duplicate column names: {dupes}",
                                  fix="Rename columns: df.columns = [...]")

        tech_cols = []
        for col in df.columns:
            for pat in TECHNICAL_COLUMN_PATTERNS:
                if re.match(pat, str(col).lower()):
                    tech_cols.append(col)
                    break
        if tech_cols:
            self.report.add_issue(section, "WARNING",
                                  f"Technical / service columns with no analytical value: {tech_cols}",
                                  fix=f"Remove them: df.drop(columns={tech_cols}, inplace=True)")

        if self.expected_columns:
            missing = [c for c in self.expected_columns if c not in df.columns]
            if missing:
                self.report.add_issue(section, "CRITICAL",
                                      f"Expected columns not present: {missing}",
                                      fix="Check the data source or update the column list.")
            else:
                self.report.add_issue(section, "OK", "All expected columns are present.")

        if not empty_cols and not dupes and not unnamed and not tech_cols:
            self.report.add_issue(section, "OK", "Dataset structure looks correct.")

    # ── 2. Data types ────────────────────────────────────────────────────────

    def check_types(self):
        df = self.df
        section = "2. DATA TYPES"
        problems = []
        type_lines = []

        for col in df.columns:
            series = df[col]
            dtype  = str(series.dtype)
            sample = series.dropna().head(5).tolist()

            # Date stored as string
            if _is_text_dtype(series) and _matches_any(str(col), DATE_KEYWORDS):
                _, ratio, fmt = _try_parse_dates(series)
                if ratio > 0.7:
                    problems.append((col, "WARNING",
                        f"Column '{col}' looks like a date but dtype is {dtype}. "
                        f"Detected format: {fmt}, success: {ratio:.0%}",
                        f"df['{col}'] = pd.to_datetime(df['{col}'], format='{fmt}', errors='coerce')"))
                else:
                    problems.append((col, "CRITICAL",
                        f"Column '{col}' may be a date but parsing failed (<70%). Sample: {sample[:3]}",
                        f"Try: pd.to_datetime(df['{col}'], errors='coerce')"))

            # Numeric stored as string
            elif _is_text_dtype(series) and _is_numeric_keyword(str(col)):
                clean = (series.dropna().astype(str)
                         .str.replace(r"[%,$€£\s]", "", regex=True)
                         .str.replace(",", "."))
                ratio = pd.to_numeric(clean, errors="coerce").notna().mean()
                if ratio > 0.7:
                    problems.append((col, "WARNING",
                        f"Column '{col}' is numeric but stored as {dtype}. "
                        f"Sample: {sample[:3]}. Conversion success: {ratio:.0%}",
                        f"df['{col}'] = pd.to_numeric(df['{col}'].astype(str)"
                        f".str.replace(r'[^0-9.]', '', regex=True), errors='coerce')"))

            # Mixed types
            if _is_text_dtype(series):
                num_ratio = pd.to_numeric(series.dropna().head(500), errors="coerce").notna().mean()
                if 0.1 < num_ratio < 0.9:
                    problems.append((col, "WARNING",
                        f"Column '{col}' has mixed types (~{num_ratio:.0%} numeric). Sample: {sample[:5]}",
                        "Split or clean the column. Numeric and non-numeric values should not be mixed."))

            type_lines.append(f"  {str(col):<35} {dtype}")

        self.report.add_issue(section, "INFO",
                              "Column types:\n" + "\n".join(type_lines))
        for col, sev, msg, fix in problems:
            self.report.add_issue(section, sev, msg, fix=fix)
        if not problems:
            self.report.add_issue(section, "OK", "Data types look correct — no obvious issues.")

    # ── 3. Missing values ────────────────────────────────────────────────────

    def check_nulls(self):
        df = self.df
        section = "3. MISSING VALUES"
        total = len(df)

        null_counts = df.isnull().sum()
        has_nulls   = null_counts[null_counts > 0]

        if not has_nulls.empty:
            lines = []
            for col, cnt in has_nulls.items():
                flag = "  <- CRITICAL FIELD" if _is_critical_field(str(col)) else ""
                lines.append(f"  {str(col):<35} {cnt:>6} ({_pct(cnt, total)}){flag}")
            self.report.add_issue(
                section, "WARNING",
                f"Nulls found in {len(has_nulls)} column(s):",
                fix="Choose a strategy: dropna / fillna(median|mode|0) / flag with a boolean column.",
                details="\n".join(lines)
            )

        for col in df.columns:
            if _is_critical_field(str(col)) and df[col].isnull().mean() > 0.3:
                ratio = df[col].isnull().mean()
                self.report.add_issue(
                    section, "CRITICAL",
                    f"Critical field '{col}' has {ratio:.0%} missing values.",
                    fix=f"Restore from source or: df.dropna(subset=['{col}'], inplace=True)")

        row_null_ratio = df.isnull().mean(axis=1)
        mostly_empty   = (row_null_ratio > 0.5).sum()
        if mostly_empty > 0:
            self.report.add_issue(
                section, "WARNING",
                f"Rows where >50% of fields are empty: {mostly_empty} ({_pct(mostly_empty, total)})",
                fix="df = df[df.isnull().mean(axis=1) <= 0.5]")

        pseudo_found = {}
        for col in df.columns:
            if _is_text_dtype(df[col]):
                vals = df[col].dropna().astype(str).str.strip().str.lower()
                cnt  = vals.isin(PSEUDO_NULL_VALUES).sum()
                if cnt > 0:
                    pseudo_found[col] = cnt

        if pseudo_found:
            details = "\n".join(f"  {c:<35} {n}" for c, n in pseudo_found.items())
            self.report.add_issue(
                section, "WARNING",
                "Text pseudo-nulls found ('N/A', 'Unknown', '-', 'null', etc.):",
                fix="df.replace(['N/A','Unknown','-','null'], np.nan, inplace=True)",
                details=details)

        if has_nulls.empty and not pseudo_found:
            self.report.add_issue(section, "OK", "No missing values detected.")

    # ── 4. Duplicates ────────────────────────────────────────────────────────

    def check_duplicates(self):
        df = self.df
        section = "4. DUPLICATES"

        full_dupes = df.duplicated().sum()
        if full_dupes > 0:
            self.report.add_issue(section, "CRITICAL",
                                  f"Fully duplicate rows: {full_dupes} ({_pct(full_dupes, len(df))})",
                                  fix="df.drop_duplicates(inplace=True)")
        else:
            self.report.add_issue(section, "OK", "No fully duplicate rows found.")

        id_cols = [c for c in df.columns if _is_id_column(str(c))]
        for col in id_cols:
            dupes = df[col].dropna().duplicated().sum()
            if dupes > 0:
                sev = "CRITICAL" if any(kw in str(col).lower() for kw in ["transaction", "order"]) \
                      else "WARNING"
                self.report.add_issue(
                    section, sev,
                    f"Duplicate values in '{col}': {dupes} ({_pct(dupes, df[col].notna().sum())})",
                    fix=f"df[df['{col}'].duplicated(keep=False)].sort_values('{col}')")
            else:
                self.report.add_issue(section, "OK", f"Column '{col}': IDs are unique.")

        date_cols     = [c for c in df.columns if _matches_any(str(c), DATE_KEYWORDS)]
        amount_cols   = [c for c in df.columns if _matches_any(str(c), ["amount", "total", "price"])]
        customer_cols = [c for c in df.columns if _matches_any(str(c), ["customer", "client"])]
        product_cols  = [c for c in df.columns if _matches_any(str(c), ["product"])]
        soft_key = (customer_cols[:1] + product_cols[:1] + date_cols[:1] + amount_cols[:1])
        soft_key = [c for c in soft_key if c in df.columns]
        if len(soft_key) >= 3:
            soft_dupes = df.duplicated(subset=soft_key).sum()
            if soft_dupes > 0:
                self.report.add_issue(
                    section, "WARNING",
                    f"Possible duplicate transactions (same {soft_key}): {soft_dupes}",
                    fix=f"df[df.duplicated(subset={soft_key}, keep=False)]")

    # ── 5. Numeric values ────────────────────────────────────────────────────

    def check_numerics(self):
        df = self.df
        section = "5. NUMERIC VALUES"
        found_issues = False

        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            neg_fields = ["quantity", "qty", "price", "amount", "total", "revenue", "cost"]
            if _matches_any(str(col), neg_fields):
                neg_cnt = (series < 0).sum()
                if neg_cnt > 0:
                    found_issues = True
                    self.report.add_issue(section, "CRITICAL",
                                          f"'{col}': {neg_cnt} negative values (min: {series.min():.2f})",
                                          fix=f"df = df[df['{col}'] >= 0]")

            if _matches_any(str(col), ["price", "amount", "total"]):
                zero_cnt = (series == 0).sum()
                if zero_cnt > 0:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                          f"'{col}': {zero_cnt} zero values",
                                          fix=f"df[df['{col}'] == 0]  — are these errors or valid data?")

            if _matches_any(str(col), ["discount", "rate", "percent"]):
                out_rng = ((series < 0) | (series > 100)).sum()
                if out_rng > 0:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                          f"'{col}': {out_rng} values outside [0, 100]",
                                          fix="Check units — percentage vs fraction. Multiply/divide by 100 if needed.")

            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)).sum()
                if outliers > 0 and outliers / len(series) > 0.01:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                          f"'{col}': {outliers} extreme outliers (3xIQR). "
                                          f"Range [{series.min():.2f}, {series.max():.2f}], "
                                          f"median {series.median():.2f}",
                                          fix=f"Inspect: df[df['{col}'] > {q3 + 3*iqr:.2f}]")

        # Cross-column consistency
        qty_col   = next((c for c in df.columns if _matches_any(str(c), ["quantity", "qty"])), None)
        price_col = next((c for c in df.columns if _matches_any(str(c), ["price"])), None)
        total_col = next((c for c in df.columns if _matches_any(str(c), ["total", "amount"])), None)

        if qty_col and price_col and total_col:
            try:
                disc_col = next((c for c in df.columns if _matches_any(str(c), ["discount"])), None)
                if disc_col:
                    sub  = df[[qty_col, price_col, total_col, disc_col]].dropna()
                    disc = pd.to_numeric(sub[disc_col], errors="coerce")
                    if disc.max() > 1:
                        disc = disc / 100
                    exp = pd.to_numeric(sub[qty_col], errors="coerce") * \
                          pd.to_numeric(sub[price_col], errors="coerce") * (1 - disc)
                else:
                    sub = df[[qty_col, price_col, total_col]].dropna()
                    exp = pd.to_numeric(sub[qty_col], errors="coerce") * \
                          pd.to_numeric(sub[price_col], errors="coerce")

                actual   = pd.to_numeric(sub[total_col], errors="coerce")
                mismatch = ((actual - exp).abs() / (exp.abs() + 1e-9) > 0.05).sum()
                if mismatch > 0:
                    found_issues = True
                    disc_part = f" x (1-{disc_col})" if disc_col else ""
                    self.report.add_issue(section, "WARNING",
                                          f"Logical inconsistency: {mismatch} rows where "
                                          f"{total_col} != {qty_col} x {price_col}{disc_part} (>5% deviation)",
                                          fix="Check amount formula. Taxes, rounding, or discount logic may differ.")
            except Exception:
                pass

        if not found_issues:
            self.report.add_issue(section, "OK", "Numeric values are within acceptable ranges.")

    # ── 6. Categorical data ──────────────────────────────────────────────────

    def check_categoricals(self):
        df = self.df
        section = "6. CATEGORICAL DATA"
        found_issues = False

        cat_cols = [c for c in df.columns if _is_category_column(str(c), df[c])]

        for col in cat_cols:
            series      = df[col].dropna().astype(str)
            unique_vals = series.unique()
            nunique     = len(unique_vals)

            lower_unique = series.str.strip().str.lower().nunique()
            if lower_unique < nunique:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                      f"'{col}': {nunique - lower_unique} duplicate(s) due to case or whitespace",
                                      fix=f"df['{col}'] = df['{col}'].str.strip().str.lower()")

            pseudo_cnt = series.str.strip().str.lower().isin(PSEUDO_NULL_VALUES).sum()
            if pseudo_cnt > 0:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                      f"'{col}': {pseudo_cnt} junk values ('Unknown', '-', etc.)",
                                      fix=f"df['{col}'].replace([...], np.nan, inplace=True)")

            if nunique > 100:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                      f"'{col}': high cardinality — {nunique} unique values.",
                                      fix="Is this really a category, or an ID / free-text field?")
            elif nunique <= 30:
                vals_str = ", ".join(sorted(str(v) for v in unique_vals[:20]))
                suffix   = f" ... (+{nunique - 20} more)" if nunique > 20 else ""
                self.report.add_issue(section, "INFO",
                                      f"'{col}' ({nunique} values): {vals_str}{suffix}")

        if not cat_cols:
            self.report.add_issue(section, "INFO", "No categorical columns detected automatically.")
        elif not found_issues:
            self.report.add_issue(section, "OK", "Categorical data looks clean.")

    # ── 7. Dates ─────────────────────────────────────────────────────────────

    def check_dates(self):
        df = self.df
        section = "7. DATES & TIME RANGE"
        now = pd.Timestamp.now()
        found_issues = False

        date_cols = [c for c in df.columns
                     if pd.api.types.is_datetime64_any_dtype(df[c])
                     or _matches_any(str(c), DATE_KEYWORDS)]

        for col in date_cols:
            series = df[col].copy()

            # Parse string columns to datetime
            if not pd.api.types.is_datetime64_any_dtype(series):
                _, ratio, fmt = _try_parse_dates(series)
                if ratio < 0.8:
                    found_issues = True
                    self.report.add_issue(section, "CRITICAL",
                                          f"'{col}': only {ratio:.0%} of values parse as dates.",
                                          fix=f"pd.to_datetime(df['{col}'], dayfirst=True, errors='coerce')")
                    continue
                series = pd.to_datetime(series, errors="coerce")
                failed = int(series.isnull().sum()) - int(df[col].isnull().sum())
                if failed > 0:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                          f"'{col}': {failed} values could not be parsed as dates.",
                                          fix="Check for non-standard formats in those rows.")

            valid_dates = series.dropna()
            if len(valid_dates) == 0:
                continue

            min_ts = valid_dates.min()
            max_ts = valid_dates.max()

            self.report.add_issue(section, "INFO",
                                  f"'{col}': range {_safe_date_str(min_ts)} -> "
                                  f"{_safe_date_str(max_ts)} "
                                  f"({_safe_days_between(max_ts, min_ts)} days)")

            # Future dates
            try:
                future = int((valid_dates > now).sum())
                if future > 0:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                          f"'{col}': {future} dates are in the future.",
                                          fix=f"df[df['{col}'] > pd.Timestamp.now()]")
            except Exception:
                pass

            # Very old dates
            try:
                old_threshold = pd.Timestamp("1990-01-01")
                very_old = int((valid_dates < old_threshold).sum())
                if very_old > 0:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                          f"'{col}': {very_old} dates before 1990.",
                                          fix=f"df[df['{col}'] < '1990-01-01']")
            except Exception:
                pass

            # Ambiguous date format
            raw_sample = df[col].dropna().astype(str).head(10).tolist()
            ambiguous  = [v for v in raw_sample if re.match(r"^\d{2}/\d{2}/\d{4}$", str(v))]
            if ambiguous:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                      f"'{col}': ambiguous format (dd/mm vs mm/dd). Samples: {ambiguous[:3]}",
                                      fix=f"pd.to_datetime(df['{col}'], dayfirst=True)")

            # Sparse coverage
            if len(valid_dates) > 30:
                try:
                    all_days     = pd.date_range(min_ts, max_ts, freq="D")
                    unique_days  = valid_dates.dt.normalize().nunique()
                    coverage     = unique_days / max(len(all_days), 1)
                    if coverage < 0.5:
                        found_issues = True
                        self.report.add_issue(section, "WARNING",
                                              f"'{col}': low date density — {unique_days} unique days "
                                              f"out of {len(all_days)} possible ({coverage:.0%}). "
                                              "Possible data gaps or incomplete export.",
                                              fix="Check that all time periods were exported from the source.")
                except Exception:
                    pass

        if not date_cols:
            self.report.add_issue(section, "INFO",
                                  "No date columns detected. If dates exist, check their dtype.")
        elif not found_issues:
            self.report.add_issue(section, "OK", "No date issues detected.")

    # ── 8. Keys & IDs ────────────────────────────────────────────────────────

    def check_keys(self):
        df = self.df
        section = "8. KEYS & IDENTIFIERS"
        found_issues = False

        id_cols = [c for c in df.columns if _is_id_column(str(c))]

        for col in id_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            null_cnt = int(df[col].isnull().sum())
            if null_cnt > 0:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                      f"'{col}': {null_cnt} empty ID value(s)",
                                      fix=f"df[df['{col}'].isnull()]")

            lengths = series.astype(str).str.len().value_counts()
            if len(lengths) > 3:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                      f"'{col}': inconsistent ID lengths — {dict(lengths.head(5))}",
                                      fix="Check for formatting errors or mixed data sources.")

            try:
                num_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
                if 0.05 < num_ratio < 0.95:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                          f"'{col}': mixed ID types ({num_ratio:.0%} numeric)",
                                          fix=f"df['{col}'] = df['{col}'].astype(str)")
            except Exception:
                pass

            cat_cols = [c for c in df.columns
                        if _is_category_column(str(c), df[c]) and c != col]
            for cat_col in cat_cols[:2]:
                try:
                    mapping      = df.groupby(col)[cat_col].nunique()
                    inconsistent = int((mapping > 1).sum())
                    if inconsistent > 0:
                        found_issues = True
                        self.report.add_issue(section, "WARNING",
                                              f"'{col}': {inconsistent} ID(s) map to multiple '{cat_col}' values.",
                                              fix=f"df.groupby('{col}')['{cat_col}'].nunique().sort_values(ascending=False).head(10)")
                except Exception:
                    pass

        if not id_cols:
            self.report.add_issue(section, "INFO",
                                  "No identifier columns detected automatically.")
        elif not found_issues:
            self.report.add_issue(section, "OK", "Keys and identifiers look consistent.")

    # ── 9. Business logic ────────────────────────────────────────────────────

    def check_business_logic(self):
        df = self.df
        section = "9. BUSINESS LOGIC & CONSISTENCY"
        found_issues = False

        disc_col  = next((c for c in df.columns if _matches_any(str(c), ["discount"])), None)
        total_col = next((c for c in df.columns if _matches_any(str(c), ["total", "amount"])), None)
        price_col = next((c for c in df.columns if _matches_any(str(c), ["price"])), None)

        if disc_col and total_col and price_col:
            try:
                sub       = df[[disc_col, total_col, price_col]].dropna()
                disc_num  = pd.to_numeric(sub[disc_col],  errors="coerce")
                total_num = pd.to_numeric(sub[total_col], errors="coerce")
                price_num = pd.to_numeric(sub[price_col], errors="coerce")
                bad = int(((disc_num > 0) & (total_num > price_num)).sum())
                if bad > 0:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                          f"{bad} rows where {total_col} > {price_col} despite a non-zero discount.",
                                          fix="Check the amount calculation logic.")
            except Exception:
                pass

        payment_col = next((c for c in df.columns
                            if _matches_any(str(c), ["payment", "pay_method"])), None)
        if payment_col:
            uniq = df[payment_col].dropna().astype(str).str.strip().str.lower().unique()
            bad  = [v for v in uniq if v in PSEUDO_NULL_VALUES or len(v) > 50]
            if bad:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                      f"'{payment_col}': suspicious values: {bad[:5]}",
                                      fix="Standardise: 'Credit Card', 'Cash', 'Online', etc.")

        prod_id_col = next((c for c in df.columns
                            if "product" in str(c).lower() and "id" in str(c).lower()), None)
        cat_col = next((c for c in df.columns if _matches_any(str(c), ["category"])), None)
        if prod_id_col and cat_col:
            try:
                mapping   = df.groupby(prod_id_col)[cat_col].nunique()
                multi_cat = int((mapping > 1).sum())
                if multi_cat > 0:
                    found_issues = True
                    self.report.add_issue(section, "CRITICAL",
                                          f"{multi_cat} product ID(s) linked to multiple categories — referential integrity issue.",
                                          fix=f"df.groupby('{prod_id_col}')['{cat_col}'].nunique().sort_values(ascending=False).head()")
            except Exception:
                pass

        if not found_issues:
            self.report.add_issue(section, "OK", "No business logic violations detected.")

    # ── Run all checks ───────────────────────────────────────────────────────

    def run(self, output_path: str = None) -> str:
        print("Starting dataset validation...")

        if not self._load():
            html = self.report.render_html()
            _open_html(html, output_path)
            return html

        steps = [
            ("[1/9] Structure",          self.check_structure),
            ("[2/9] Data types",         self.check_types),
            ("[3/9] Missing values",     self.check_nulls),
            ("[4/9] Duplicates",         self.check_duplicates),
            ("[5/9] Numeric values",     self.check_numerics),
            ("[6/9] Categorical data",   self.check_categoricals),
            ("[7/9] Dates",              self.check_dates),
            ("[8/9] Keys & identifiers", self.check_keys),
            ("[9/9] Business logic",     self.check_business_logic),
        ]
        for label, fn in steps:
            print(f"  {label} ...")
            try:
                fn()
            except Exception as exc:
                self.report.add_issue(
                    label.split("]")[1].strip().upper(), "WARNING",
                    f"Check raised an unexpected error: {exc}",
                    fix="Please report this issue with your dataset sample."
                )

        c = self.report.summary_counts
        print(f"\nValidation complete.")
        print(f"  Critical: {c.get('CRITICAL',0)}  "
              f"Warnings: {c.get('WARNING',0)}  "
              f"Info: {c.get('INFO',0)}  "
              f"OK: {c.get('OK',0)}")

        html = self.report.render_html()
        _open_html(html, output_path)
        return html


# ─────────────────────────────────────────────
#  HTML output helper
# ─────────────────────────────────────────────

def _open_html(html: str, output_path: str = None):
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        path = output_path
        print(f"\nReport saved to: {path}")
    else:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        )
        tmp.write(html)
        tmp.close()
        path = tmp.name

    webbrowser.open(f"file://{os.path.abspath(path)}")
    print(f"Report opened in browser: {path}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Validation Agent — validates a CSV before data analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_validator.py data.csv
  python dataset_validator.py data.csv --columns order_id customer_id price quantity
  python dataset_validator.py data.csv --output report.html
        """
    )
    parser.add_argument("filepath", help="Path to the CSV file")
    parser.add_argument("--columns", nargs="*", default=[],
                        help="Expected column names (optional)")
    parser.add_argument("--output", default=None,
                        help="Save HTML report to this path (optional). "
                             "If omitted, a temp file is created and opened in the browser.")
    args = parser.parse_args()

    agent = DatasetValidationAgent(
        filepath=args.filepath,
        expected_columns=args.columns,
    )
    agent.run(output_path=args.output)


if __name__ == "__main__":
    main()
