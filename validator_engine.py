"""
Validator Engine
================
Core validation logic extracted from dataset_validator.py.
Returns structured issue objects (dicts) instead of rendering HTML directly.
Each issue carries optional `fixes` — a list of available automated actions
the web UI can invoke via the /api/fix endpoint.
"""

import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Constants  (same as original)
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
    if dtype_str in ("object", "string"):
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
    return series.dtype == object or str(series.dtype) == "string"

def _try_parse_dates(series: pd.Series):
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

    try:
        parsed_inf = pd.to_datetime(sample, errors="coerce")
        ratio_inf = parsed_inf.notna().mean()
        if ratio_inf > best_ratio:
            best_ratio, best_parsed, best_fmt = ratio_inf, parsed_inf, "auto-inferred"
    except Exception:
        pass

    return best_parsed, best_ratio, best_fmt

def _safe_date_str(ts) -> str:
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
#  Issue builder
# ─────────────────────────────────────────────

def _make_issue(section: str, severity: str, message: str,
                fix_text: str = "", details: str = "",
                fixes: list = None) -> dict:
    """
    Build a serialisable issue dict.
    `fixes` is a list of dicts: {label, fix_id, params}
    representing automated actions the UI can call.
    """
    return {
        "id": str(uuid.uuid4()),
        "section": section,
        "severity": severity,
        "message": message,
        "fix_text": fix_text,      # human-readable suggestion
        "details": details,
        "fixes": fixes or [],      # automated fix actions
    }


# ─────────────────────────────────────────────
#  Validation Engine
# ─────────────────────────────────────────────

class DatasetValidationEngine:

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df: pd.DataFrame = None
        self.issues: list = []
        self.summary_counts = {"CRITICAL": 0, "WARNING": 0, "INFO": 0, "OK": 0}

    def _add(self, section, severity, message, fix_text="", details="", fixes=None):
        issue = _make_issue(section, severity, message, fix_text, details, fixes)
        self.issues.append(issue)
        self.summary_counts[severity] = self.summary_counts.get(severity, 0) + 1

    # ── Load ──────────────────────────────────────────────────

    def load(self) -> bool:
        encodings  = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]
        separators = [",", ";", "\t", "|"]
        path = Path(self.filepath)

        if not path.exists():
            self._add("1. Dataset Structure", "CRITICAL",
                      f"File not found: {self.filepath}",
                      fix_text="Check the file path.")
            return False

        if path.suffix.lower() != ".csv":
            self._add("1. Dataset Structure", "WARNING",
                      f"File extension is not .csv: {path.suffix}",
                      fix_text="Make sure the file is a valid CSV.")

        for enc in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(self.filepath, encoding=enc, sep=sep, low_memory=False)
                    if df.shape[1] > 1 or sep == ",":
                        self.df = df
                        self._add("1. Dataset Structure", "INFO",
                                  f"File loaded successfully. Encoding: {enc}, separator: '{sep}'")
                        return True
                except Exception:
                    continue

        self._add("1. Dataset Structure", "CRITICAL",
                  "Could not load the file with any standard encoding or separator.",
                  fix_text="Check file encoding (UTF-8 recommended) and separator.")
        return False

    def run_all_checks(self):
        for fn in [
            self.check_structure,
            self.check_types,
            self.check_nulls,
            self.check_duplicates,
            self.check_numerics,
            self.check_categoricals,
            self.check_dates,
            self.check_keys,
            self.check_business_logic,
        ]:
            try:
                fn()
            except Exception as exc:
                self._add("Error", "WARNING",
                          f"Check {fn.__name__} raised an error: {exc}")

    # ── 1. Structure ──────────────────────────────────────────

    def check_structure(self):
        df = self.df
        sec = "1. Dataset Structure"

        self._add(sec, "INFO",
                  f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        if df.empty:
            self._add(sec, "CRITICAL", "Dataset is empty (0 rows).",
                      fix_text="Check the data source.")
            return

        if df.shape[0] < 5:
            self._add(sec, "WARNING",
                      f"Very few rows: {df.shape[0]}. Analysis may be unreliable.")

        unnamed = [c for c in df.columns if str(c).strip() == ""]
        if unnamed:
            self._add(sec, "CRITICAL",
                      f"{len(unnamed)} column(s) have no name (blank headers).",
                      fix_text="Add headers or remove extra separator characters.")

        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            self._add(sec, "WARNING",
                      f"Fully empty columns detected: {empty_cols}",
                      fix_text="Remove these columns — they carry no information.",
                      fixes=[{"label": "Drop empty columns",
                               "fix_id": "drop_empty_columns", "params": {}}])

        dupes = list({c for c in df.columns if list(df.columns).count(c) > 1})
        if dupes:
            self._add(sec, "CRITICAL",
                      f"Duplicate column names: {dupes}",
                      fix_text="Rename or deduplicate columns.",
                      fixes=[{"label": "Keep first occurrence, remove duplicates",
                               "fix_id": "drop_full_duplicate_columns", "params": {}}])

        tech_cols = []
        for col in df.columns:
            for pat in TECHNICAL_COLUMN_PATTERNS:
                if re.match(pat, str(col).lower()):
                    tech_cols.append(col)
                    break
        if tech_cols:
            self._add(sec, "WARNING",
                      f"Technical/service columns with no analytical value: {tech_cols}",
                      fix_text="Drop these columns.",
                      fixes=[{"label": f"Drop '{c}'", "fix_id": "drop_column", "params": {"col": c}}
                             for c in tech_cols])

        if not empty_cols and not dupes and not unnamed and not tech_cols:
            self._add(sec, "OK", "Dataset structure looks correct.")

    # ── 2. Data types ─────────────────────────────────────────

    def check_types(self):
        df = self.df
        sec = "2. Data Types"
        problems = []

        type_lines = [f"{'Column':<35} {'Type'}" ,
                      "─" * 50]
        for col in df.columns:
            type_lines.append(f"{str(col):<35} {str(df[col].dtype)}")

        self._add(sec, "INFO", "Detected column types:",
                  details="\n".join(type_lines))

        for col in df.columns:
            series = df[col]
            dtype  = str(series.dtype)
            sample = series.dropna().head(5).tolist()

            # Date stored as string
            if _is_text_dtype(series) and _matches_any(str(col), DATE_KEYWORDS):
                _, ratio, fmt = _try_parse_dates(series)
                if ratio > 0.7:
                    problems.append(self._add(
                        sec, "WARNING",
                        f"'{col}' looks like a date but is stored as text. "
                        f"Detected format: {fmt} (success: {ratio:.0%})",
                        fix_text=f"Convert with pd.to_datetime(df['{col}'], format='{fmt}', errors='coerce')",
                        fixes=[{"label": f"Convert '{col}' to datetime",
                                "fix_id": "cast_to_datetime",
                                "params": {"col": col, "format": fmt}}]))
                else:
                    problems.append(self._add(
                        sec, "CRITICAL",
                        f"'{col}' may be a date but auto-parsing failed (<70%). Sample: {sample[:3]}",
                        fix_text=f"Try: pd.to_datetime(df['{col}'], errors='coerce')",
                        fixes=[{"label": f"Attempt datetime conversion for '{col}'",
                                "fix_id": "cast_to_datetime",
                                "params": {"col": col, "format": "auto-inferred"}}]))

            # Numeric stored as string
            elif _is_text_dtype(series) and _is_numeric_keyword(str(col)):
                clean = (series.dropna().astype(str)
                         .str.replace(r"[%,$€£\s]", "", regex=True)
                         .str.replace(",", "."))
                ratio = pd.to_numeric(clean, errors="coerce").notna().mean()
                if ratio > 0.7:
                    problems.append(self._add(
                        sec, "WARNING",
                        f"'{col}' appears numeric but is stored as text ({dtype}). "
                        f"Conversion success: {ratio:.0%}. Sample: {sample[:3]}",
                        fix_text="Strip currency symbols and cast to numeric.",
                        fixes=[{"label": f"Convert '{col}' to numeric",
                                "fix_id": "cast_to_numeric",
                                "params": {"col": col}}]))

            # Mixed types
            if _is_text_dtype(series):
                num_ratio = pd.to_numeric(series.dropna().head(500), errors="coerce").notna().mean()
                if 0.1 < num_ratio < 0.9:
                    problems.append(self._add(
                        sec, "WARNING",
                        f"'{col}' has mixed types (~{num_ratio:.0%} numeric). Sample: {sample[:5]}",
                        fix_text="Split or clean the column — numeric and text should not be mixed.",
                        fixes=[{"label": f"Cast '{col}' to string (keep all as text)",
                                "fix_id": "cast_to_string",
                                "params": {"col": col}},
                               {"label": f"Cast '{col}' to numeric (non-parseable → NaN)",
                                "fix_id": "cast_to_numeric",
                                "params": {"col": col}}]))

        if not problems:
            self._add(sec, "OK", "Data types look correct — no obvious issues.")

    # ── 3. Missing values ─────────────────────────────────────

    def check_nulls(self):
        df = self.df
        sec = "3. Missing Values"
        total = len(df)

        null_counts = df.isnull().sum()
        has_nulls   = null_counts[null_counts > 0]

        if not has_nulls.empty:
            lines = []
            for col, cnt in has_nulls.items():
                flag = "  ← CRITICAL FIELD" if _is_critical_field(str(col)) else ""
                lines.append(f"{str(col):<35} {cnt:>6} ({_pct(cnt, total)}){flag}")

            self._add(sec, "WARNING",
                      f"Null values found in {len(has_nulls)} column(s):",
                      fix_text="Choose a fill strategy per column: mean/median for numeric, mode for categorical, or drop rows.",
                      details="\n".join(lines))

        for col, cnt in has_nulls.items():
            ratio = cnt / total
            fixes = []
            col_series = df[col]

            is_numeric = pd.api.types.is_numeric_dtype(col_series)
            is_text    = _is_text_dtype(col_series)

            if is_numeric:
                fixes += [
                    {"label": f"Fill with mean ({col_series.mean():.2f})",
                     "fix_id": "fill_mean", "params": {"col": col}},
                    {"label": f"Fill with median ({col_series.median():.2f})",
                     "fix_id": "fill_median", "params": {"col": col}},
                    {"label": "Fill with 0",
                     "fix_id": "fill_zero", "params": {"col": col}},
                ]
            if is_text:
                fixes += [
                    {"label": "Fill with mode (most frequent value)",
                     "fix_id": "fill_mode", "params": {"col": col}},
                    {"label": "Fill with 'Unknown'",
                     "fix_id": "fill_unknown", "params": {"col": col}},
                ]

            fixes.append({"label": f"Drop rows where '{col}' is null",
                          "fix_id": "drop_null_rows", "params": {"col": col}})

            if _is_critical_field(str(col)) and ratio > 0.3:
                self._add(sec, "CRITICAL",
                          f"Critical field '{col}' has {ratio:.0%} missing values ({cnt:,} rows).",
                          fix_text="Restore from source, or drop rows / fill with a default.",
                          fixes=fixes)
            elif ratio > 0:
                self._add(sec, "WARNING",
                          f"'{col}': {cnt:,} null values ({ratio:.1%})",
                          fix_text="Choose an appropriate fill or drop strategy.",
                          fixes=fixes)

        # Mostly-empty rows
        row_null_ratio = df.isnull().mean(axis=1)
        mostly_empty   = (row_null_ratio > 0.5).sum()
        if mostly_empty > 0:
            self._add(sec, "WARNING",
                      f"{mostly_empty} row(s) have >50% empty fields ({_pct(mostly_empty, total)}).",
                      fix_text="Consider removing these near-empty rows.")

        # Pseudo-nulls in text columns
        pseudo_found = {}
        for col in df.columns:
            if _is_text_dtype(df[col]):
                vals = df[col].dropna().astype(str).str.strip().str.lower()
                cnt  = vals.isin(PSEUDO_NULL_VALUES).sum()
                if cnt > 0:
                    pseudo_found[col] = cnt

        if pseudo_found:
            details = "\n".join(f"{c:<35} {n}" for c, n in pseudo_found.items())
            self._add(sec, "WARNING",
                      "Text pseudo-nulls detected ('N/A', 'Unknown', '-', 'null', etc.):",
                      fix_text="Replace these with proper NaN values.",
                      details=details,
                      fixes=[{"label": f"Replace pseudo-nulls in '{c}' with NaN",
                               "fix_id": "replace_pseudo_nulls", "params": {"col": c}}
                             for c in pseudo_found])

        if has_nulls.empty and not pseudo_found:
            self._add(sec, "OK", "No missing values detected.")

    # ── 4. Duplicates ─────────────────────────────────────────

    def check_duplicates(self):
        df = self.df
        sec = "4. Duplicates"

        full_dupes = df.duplicated().sum()
        if full_dupes > 0:
            self._add(sec, "CRITICAL",
                      f"Fully duplicate rows: {full_dupes:,} ({_pct(full_dupes, len(df))})",
                      fix_text="Remove exact duplicate rows.",
                      fixes=[{"label": "Remove all fully duplicate rows",
                               "fix_id": "drop_full_duplicates", "params": {}}])
        else:
            self._add(sec, "OK", "No fully duplicate rows found.")

        id_cols = [c for c in df.columns if _is_id_column(str(c))]
        for col in id_cols:
            dupes = df[col].dropna().duplicated().sum()
            if dupes > 0:
                sev = "CRITICAL" if any(kw in str(col).lower()
                                        for kw in ["transaction", "order"]) else "WARNING"
                self._add(sec, sev,
                          f"Duplicate values in ID column '{col}': {dupes:,} ({_pct(dupes, df[col].notna().sum())})",
                          fix_text=f"Inspect duplicates or drop them keeping first occurrence.",
                          fixes=[{"label": f"Drop duplicate rows by '{col}' (keep first)",
                                  "fix_id": "drop_col_duplicates", "params": {"col": col}}])
            else:
                self._add(sec, "OK", f"'{col}': all ID values are unique.")

    # ── 5. Numeric values ─────────────────────────────────────

    def check_numerics(self):
        df = self.df
        sec = "5. Numeric Values"
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
                    self._add(sec, "CRITICAL",
                              f"'{col}': {neg_cnt:,} negative values (min: {series.min():.2f})",
                              fix_text="Remove or convert negative rows.",
                              fixes=[
                                  {"label": f"Remove rows where '{col}' < 0",
                                   "fix_id": "drop_negative_rows", "params": {"col": col}},
                                  {"label": f"Convert negatives to absolute values in '{col}'",
                                   "fix_id": "abs_negative", "params": {"col": col}},
                              ])

            if _matches_any(str(col), ["price", "amount", "total"]):
                zero_cnt = (series == 0).sum()
                if zero_cnt > 0:
                    found_issues = True
                    self._add(sec, "WARNING",
                              f"'{col}': {zero_cnt:,} zero values — verify these are intentional.",
                              fix_text="Decide: are zeroes valid records or data errors?",
                              fixes=[{"label": f"Remove rows where '{col}' == 0",
                                      "fix_id": "drop_zero_rows", "params": {"col": col}}])

            if _matches_any(str(col), ["discount", "rate", "percent"]):
                out_rng = ((series < 0) | (series > 100)).sum()
                if out_rng > 0:
                    found_issues = True
                    self._add(sec, "WARNING",
                              f"'{col}': {out_rng:,} values outside [0, 100] — wrong units?",
                              fix_text="Check units. If stored as fraction (0–1), multiply by 100.",
                              fixes=[{"label": f"Divide '{col}' by 100 (percent → fraction)",
                                      "fix_id": "percent_divide_100", "params": {"col": col}}])

            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)).sum()
                if outliers > 0 and outliers / len(series) > 0.01:
                    found_issues = True
                    self._add(sec, "WARNING",
                              f"'{col}': {outliers:,} extreme outliers (3×IQR). "
                              f"Range [{series.min():.2f}, {series.max():.2f}], median {series.median():.2f}",
                              fix_text="Inspect outliers — remove or cap them.",
                              fixes=[{"label": f"Cap outliers in '{col}' to 3×IQR bounds",
                                      "fix_id": "cap_outliers_iqr", "params": {"col": col}}])

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
                    exp = (pd.to_numeric(sub[qty_col], errors="coerce") *
                           pd.to_numeric(sub[price_col], errors="coerce") * (1 - disc))
                else:
                    sub = df[[qty_col, price_col, total_col]].dropna()
                    exp = (pd.to_numeric(sub[qty_col], errors="coerce") *
                           pd.to_numeric(sub[price_col], errors="coerce"))

                actual   = pd.to_numeric(sub[total_col], errors="coerce")
                mismatch = ((actual - exp).abs() / (exp.abs() + 1e-9) > 0.05).sum()
                if mismatch > 0:
                    found_issues = True
                    disc_part = f" × (1 − {disc_col})" if disc_col else ""
                    self._add(sec, "WARNING",
                              f"Calculation mismatch: {mismatch:,} rows where "
                              f"{total_col} ≠ {qty_col} × {price_col}{disc_part} (>5% deviation)",
                              fix_text="Check your amount formula — taxes, rounding, or discounts may differ.")
            except Exception:
                pass

        if not found_issues:
            self._add(sec, "OK", "Numeric values are within acceptable ranges.")

    # ── 6. Categorical data ───────────────────────────────────

    def check_categoricals(self):
        df = self.df
        sec = "6. Categorical Data"
        found_issues = False

        cat_cols = [c for c in df.columns if _is_category_column(str(c), df[c])]

        for col in cat_cols:
            series      = df[col].dropna().astype(str)
            unique_vals = series.unique()
            nunique     = len(unique_vals)

            lower_unique = series.str.strip().str.lower().nunique()
            if lower_unique < nunique:
                found_issues = True
                self._add(sec, "WARNING",
                          f"'{col}': {nunique - lower_unique} duplicate(s) due to inconsistent case or whitespace",
                          fix_text="Normalise values by stripping and lower-casing.",
                          fixes=[
                              {"label": f"Strip & lowercase '{col}'",
                               "fix_id": "strip_lowercase", "params": {"col": col}},
                              {"label": f"Strip & title-case '{col}'",
                               "fix_id": "strip_titlecase", "params": {"col": col}},
                          ])

            pseudo_cnt = series.str.strip().str.lower().isin(PSEUDO_NULL_VALUES).sum()
            if pseudo_cnt > 0:
                found_issues = True
                self._add(sec, "WARNING",
                          f"'{col}': {pseudo_cnt:,} junk placeholder values ('Unknown', '-', etc.)",
                          fix_text="Replace with NaN for consistent null handling.",
                          fixes=[{"label": f"Replace pseudo-nulls in '{col}' with NaN",
                                  "fix_id": "replace_pseudo_nulls", "params": {"col": col}}])

            if nunique > 100:
                found_issues = True
                self._add(sec, "WARNING",
                          f"'{col}': high cardinality — {nunique:,} unique values. Is this truly categorical?",
                          fix_text="Consider whether this should be an ID or free-text field instead.")
            elif nunique <= 30:
                vals_str = ", ".join(sorted(str(v) for v in unique_vals[:20]))
                suffix   = f" … (+{nunique - 20} more)" if nunique > 20 else ""
                self._add(sec, "INFO",
                          f"'{col}' has {nunique} unique values: {vals_str}{suffix}")

        if not cat_cols:
            self._add(sec, "INFO", "No categorical columns detected automatically.")
        elif not found_issues:
            self._add(sec, "OK", "Categorical data looks clean.")

    # ── 7. Dates ──────────────────────────────────────────────

    def check_dates(self):
        df = self.df
        sec = "7. Dates & Time Range"
        now = pd.Timestamp.now()
        found_issues = False

        date_cols = [c for c in df.columns
                     if pd.api.types.is_datetime64_any_dtype(df[c])
                     or _matches_any(str(c), DATE_KEYWORDS)]

        for col in date_cols:
            series = df[col].copy()

            if not pd.api.types.is_datetime64_any_dtype(series):
                _, ratio, fmt = _try_parse_dates(series)
                if ratio < 0.8:
                    found_issues = True
                    self._add(sec, "CRITICAL",
                              f"'{col}': only {ratio:.0%} of values parse as dates.",
                              fix_text="Attempt datetime conversion — failures become NaT.",
                              fixes=[{"label": f"Convert '{col}' to datetime",
                                      "fix_id": "cast_to_datetime",
                                      "params": {"col": col, "format": "auto-inferred"}}])
                    continue
                series = pd.to_datetime(series, errors="coerce")

            valid_dates = series.dropna()
            if len(valid_dates) == 0:
                continue

            min_ts, max_ts = valid_dates.min(), valid_dates.max()
            self._add(sec, "INFO",
                      f"'{col}': {_safe_date_str(min_ts)} → {_safe_date_str(max_ts)} "
                      f"({_safe_days_between(max_ts, min_ts)} days span)")

            try:
                future = int((valid_dates > now).sum())
                if future > 0:
                    found_issues = True
                    self._add(sec, "WARNING",
                              f"'{col}': {future:,} dates are in the future.",
                              fix_text="These may be data entry errors.",
                              fixes=[{"label": f"Remove rows with future dates in '{col}'",
                                      "fix_id": "drop_future_dates", "params": {"col": col}}])
            except Exception:
                pass

            try:
                old_threshold = pd.Timestamp("1990-01-01")
                very_old = int((valid_dates < old_threshold).sum())
                if very_old > 0:
                    found_issues = True
                    self._add(sec, "WARNING",
                              f"'{col}': {very_old:,} dates before 1990 — possibly erroneous.",
                              fix_text="Inspect these rows manually.")
            except Exception:
                pass

            raw_sample = df[col].dropna().astype(str).head(10).tolist()
            ambiguous  = [v for v in raw_sample if re.match(r"^\d{2}/\d{2}/\d{4}$", str(v))]
            if ambiguous:
                found_issues = True
                self._add(sec, "WARNING",
                          f"'{col}': ambiguous date format (dd/mm vs mm/dd). Samples: {ambiguous[:3]}",
                          fix_text="Specify dayfirst=True or False when parsing.")

            if len(valid_dates) > 30:
                try:
                    all_days    = pd.date_range(min_ts, max_ts, freq="D")
                    unique_days = valid_dates.dt.normalize().nunique()
                    coverage    = unique_days / max(len(all_days), 1)
                    if coverage < 0.5:
                        found_issues = True
                        self._add(sec, "WARNING",
                                  f"'{col}': low date density — {unique_days} unique days out of "
                                  f"{len(all_days)} possible ({coverage:.0%}). Possible data gaps.",
                                  fix_text="Check whether all time periods were exported from the source.")
                except Exception:
                    pass

        if not date_cols:
            self._add(sec, "INFO", "No date columns detected automatically.")
        elif not found_issues:
            self._add(sec, "OK", "No date issues detected.")

    # ── 8. Keys & Identifiers ─────────────────────────────────

    def check_keys(self):
        df = self.df
        sec = "8. Keys & Identifiers"
        found_issues = False

        id_cols = [c for c in df.columns if _is_id_column(str(c))]

        for col in id_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            null_cnt = int(df[col].isnull().sum())
            if null_cnt > 0:
                found_issues = True
                self._add(sec, "WARNING",
                          f"'{col}': {null_cnt:,} empty ID value(s)",
                          fix_text="IDs should not be null — inspect or drop these rows.",
                          fixes=[{"label": f"Drop rows with null '{col}'",
                                  "fix_id": "drop_null_rows", "params": {"col": col}}])

            lengths = series.astype(str).str.len().value_counts()
            if len(lengths) > 3:
                found_issues = True
                self._add(sec, "WARNING",
                          f"'{col}': inconsistent ID length — {dict(lengths.head(5))}",
                          fix_text="May indicate mixed data sources or formatting errors.")

            try:
                num_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
                if 0.05 < num_ratio < 0.95:
                    found_issues = True
                    self._add(sec, "WARNING",
                              f"'{col}': mixed ID types ({num_ratio:.0%} numeric)",
                              fix_text="Cast to string for consistent handling.",
                              fixes=[{"label": f"Cast '{col}' to string",
                                      "fix_id": "cast_to_string", "params": {"col": col}}])
            except Exception:
                pass

        if not id_cols:
            self._add(sec, "INFO", "No identifier columns detected automatically.")
        elif not found_issues:
            self._add(sec, "OK", "Keys and identifiers look consistent.")

    # ── 9. Business logic ─────────────────────────────────────

    def check_business_logic(self):
        df = self.df
        sec = "9. Business Logic"
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
                    self._add(sec, "WARNING",
                              f"{bad:,} rows: {total_col} > {price_col} despite a non-zero discount.",
                              fix_text="Check the amount calculation — discount may not be applied.")
            except Exception:
                pass

        prod_id_col = next((c for c in df.columns
                            if "product" in str(c).lower() and "id" in str(c).lower()), None)
        cat_col = next((c for c in df.columns if _matches_any(str(c), ["category"])), None)

        if prod_id_col and cat_col:
            try:
                mapping   = df.groupby(prod_id_col)[cat_col].nunique()
                multi_cat = int((mapping > 1).sum())
                if multi_cat > 0:
                    found_issues = True
                    self._add(sec, "CRITICAL",
                              f"{multi_cat:,} product ID(s) linked to multiple categories — referential integrity issue.",
                              fix_text="A single product should belong to one category. Investigate duplicates.")
            except Exception:
                pass

        if not found_issues:
            self._add(sec, "OK", "No business logic violations detected.")
