"""
Dataset Validator Web Application
==================================
Flask backend that exposes the validation engine via HTTP API.
Supports dataset upload, full validation, per-issue fix actions,
and cleaned dataset download.
"""

import os
import io
import re
import json
import uuid
import tempfile
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template

# Import the core validation engine (refactored to return structured data)
from validator_engine import DatasetValidationEngine

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit

# In-memory session store: session_id -> {"df": DataFrame, "path": str, "filename": str}
SESSIONS: dict = {}

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "dataset_validator_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Accepts a CSV file upload.
    Returns: session_id + full validation report (list of issues).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    session_id = str(uuid.uuid4())
    save_path = UPLOAD_FOLDER / f"{session_id}_{file.filename}"
    file.save(str(save_path))

    engine = DatasetValidationEngine(str(save_path))
    success = engine.load()

    if not success:
        return jsonify({
            "session_id": session_id,
            "filename": file.filename,
            "loaded": False,
            "issues": engine.issues,
            "summary": engine.summary_counts,
            "shape": None,
            "columns": [],
        })

    SESSIONS[session_id] = {
        "df": engine.df.copy(),
        "path": str(save_path),
        "filename": file.filename,
    }

    engine.run_all_checks()

    return jsonify({
        "session_id": session_id,
        "filename": file.filename,
        "loaded": True,
        "issues": engine.issues,
        "summary": engine.summary_counts,
        "shape": {"rows": engine.df.shape[0], "cols": engine.df.shape[1]},
        "columns": list(engine.df.columns),
        "dtypes": {col: str(engine.df[col].dtype) for col in engine.df.columns},
    })


@app.route("/api/fix", methods=["POST"])
def fix_issue():
    """
    Apply a single fix action to the session DataFrame.
    Body: { session_id, fix_id, params? }
    Returns: updated issues list + summary after the fix.
    """
    body = request.get_json(force=True)
    session_id = body.get("session_id")
    fix_id = body.get("fix_id")
    params = body.get("params", {})

    if session_id not in SESSIONS:
        return jsonify({"error": "Session not found or expired"}), 404

    session = SESSIONS[session_id]
    df: pd.DataFrame = session["df"]

    try:
        df, message = apply_fix(df, fix_id, params)
        session["df"] = df
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500

    # Re-run validation on the updated DataFrame
    engine = DatasetValidationEngine(session["path"])
    engine.df = df.copy()
    engine.run_all_checks()

    return jsonify({
        "success": True,
        "message": message,
        "issues": engine.issues,
        "summary": engine.summary_counts,
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    })


@app.route("/api/download/<session_id>")
def download(session_id):
    """Download the current (possibly fixed) DataFrame as CSV."""
    if session_id not in SESSIONS:
        return jsonify({"error": "Session not found"}), 404

    session = SESSIONS[session_id]
    df: pd.DataFrame = session["df"]
    original_name = Path(session["filename"]).stem

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{original_name}_cleaned.csv",
    )


@app.route("/api/preview/<session_id>")
def preview(session_id):
    """Return first 100 rows of the current DataFrame as JSON."""
    if session_id not in SESSIONS:
        return jsonify({"error": "Session not found"}), 404

    df: pd.DataFrame = SESSIONS[session_id]["df"]
    preview_df = df.head(100).copy()

    # Convert non-serialisable types to strings
    for col in preview_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        preview_df[col] = preview_df[col].astype(str)

    return jsonify({
        "columns": list(preview_df.columns),
        "rows": preview_df.where(pd.notnull(preview_df), None).values.tolist(),
        "total_rows": len(df),
    })


# ─────────────────────────────────────────────────────────────
#  Fix Actions Registry
# ─────────────────────────────────────────────────────────────

def apply_fix(df: pd.DataFrame, fix_id: str, params: dict):
    """
    Dispatch a fix_id to the corresponding transformation.
    Returns (modified_df, human_readable_message).
    """
    col = params.get("col")

    # ── Null handling ──────────────────────────────────────────
    if fix_id == "fill_mean":
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mean_val = round(df[col].mean(), 4)
        df[col] = df[col].fillna(mean_val)
        return df, f"Filled {col!r} NaN with mean ({mean_val})"

    if fix_id == "fill_median":
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_val = round(df[col].median(), 4)
        df[col] = df[col].fillna(median_val)
        return df, f"Filled {col!r} NaN with median ({median_val})"

    if fix_id == "fill_mode":
        mode_val = df[col].mode(dropna=True)
        if len(mode_val) == 0:
            return df, f"No mode found in {col!r}, nothing changed"
        df[col] = df[col].fillna(mode_val.iloc[0])
        return df, f"Filled {col!r} NaN with mode ({mode_val.iloc[0]!r})"

    if fix_id == "fill_zero":
        df[col] = df[col].fillna(0)
        return df, f"Filled {col!r} NaN with 0"

    if fix_id == "fill_unknown":
        df[col] = df[col].fillna("Unknown")
        return df, f"Filled {col!r} NaN with 'Unknown'"

    if fix_id == "drop_null_rows":
        before = len(df)
        df = df.dropna(subset=[col])
        dropped = before - len(df)
        return df, f"Dropped {dropped} rows with NaN in {col!r}"

    # ── Duplicates ────────────────────────────────────────────
    if fix_id == "drop_full_duplicates":
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        return df, f"Removed {dropped} fully duplicate rows"

    if fix_id == "drop_col_duplicates":
        before = len(df)
        df = df.drop_duplicates(subset=[col], keep="first")
        dropped = before - len(df)
        return df, f"Removed {dropped} duplicate rows (kept first) by {col!r}"

    # ── Type casting ──────────────────────────────────────────
    if fix_id == "cast_to_numeric":
        clean = (df[col].astype(str)
                 .str.replace(r"[%,$€£\s]", "", regex=True)
                 .str.replace(",", "."))
        df[col] = pd.to_numeric(clean, errors="coerce")
        return df, f"Converted {col!r} to numeric (non-parseable → NaN)"

    if fix_id == "cast_to_datetime":
        fmt = params.get("format", None)
        if fmt and fmt != "auto-inferred":
            df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
        else:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return df, f"Converted {col!r} to datetime"

    if fix_id == "cast_to_string":
        df[col] = df[col].astype(str)
        return df, f"Converted {col!r} to string"

    # ── Categorical cleaning ───────────────────────────────────
    if fix_id == "strip_lowercase":
        df[col] = df[col].astype(str).str.strip().str.lower()
        return df, f"Stripped whitespace and lowercased {col!r}"

    if fix_id == "strip_titlecase":
        df[col] = df[col].astype(str).str.strip().str.title()
        return df, f"Stripped whitespace and title-cased {col!r}"

    if fix_id == "replace_pseudo_nulls":
        PSEUDO = {"n/a", "na", "nan", "null", "none", "nil", "-", "--", "---",
                  "unknown", "missing", "empty", "", " ", "not available", "undefined"}
        mask = df[col].astype(str).str.strip().str.lower().isin(PSEUDO)
        replaced = int(mask.sum())
        df.loc[mask, col] = np.nan
        return df, f"Replaced {replaced} pseudo-null text values in {col!r} with NaN"

    # ── Numeric fixes ──────────────────────────────────────────
    if fix_id == "drop_negative_rows":
        before = len(df)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df[col].isna() | (df[col] >= 0)]
        dropped = before - len(df)
        return df, f"Removed {dropped} rows where {col!r} < 0"

    if fix_id == "abs_negative":
        df[col] = pd.to_numeric(df[col], errors="coerce").abs()
        return df, f"Converted negative values in {col!r} to absolute values"

    if fix_id == "cap_outliers_iqr":
        df[col] = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        before = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
        return df, f"Capped {before} outliers in {col!r} to [{lo:.2f}, {hi:.2f}]"

    if fix_id == "drop_zero_rows":
        before = len(df)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df[col].isna() | (df[col] != 0)]
        dropped = before - len(df)
        return df, f"Removed {dropped} rows where {col!r} == 0"

    if fix_id == "percent_divide_100":
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100
        return df, f"Divided {col!r} by 100 (converted percentage to fraction)"

    # ── Column-level fixes ────────────────────────────────────
    if fix_id == "drop_column":
        df = df.drop(columns=[col], errors="ignore")
        return df, f"Dropped column {col!r}"

    if fix_id == "drop_empty_columns":
        empty = [c for c in df.columns if df[c].isna().all()]
        df = df.drop(columns=empty, errors="ignore")
        return df, f"Dropped {len(empty)} fully empty column(s): {empty}"

    if fix_id == "drop_full_duplicate_columns":
        df = df.loc[:, ~df.columns.duplicated()]
        return df, "Removed duplicate column names (kept first occurrence)"

    # ── Date fixes ────────────────────────────────────────────
    if fix_id == "drop_future_dates":
        before = len(df)
        now = pd.Timestamp.now()
        parsed = pd.to_datetime(df[col], errors="coerce")
        df = df[parsed.isna() | (parsed <= now)]
        dropped = before - len(df)
        return df, f"Removed {dropped} rows with future dates in {col!r}"

    raise ValueError(f"Unknown fix_id: {fix_id!r}")


# ─────────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
