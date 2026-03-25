# Dataset Validator — Web Application

A web application for validating CSV datasets before data analysis.
Upload a CSV, receive a full diagnostic report across 9 check categories,
and apply one-click automated fixes directly in the browser.

---

## Features

- **9-category validation**: Structure, Types, Missing Values, Duplicates,
  Numerics, Categoricals, Dates, Keys, Business Logic
- **Severity levels**: CRITICAL / WARNING / INFO / OK
- **One-click fixes**: Each issue with an automated remedy shows action buttons.
  Clicking a fix button applies the transformation server-side and re-runs
  the full validation so the report updates immediately.
- **Live data preview**: See the first 100 rows of the current (possibly fixed)
  DataFrame at any time.
- **Download cleaned CSV**: Export the fixed dataset at any point.
- **Filter by severity**: View only Critical, Warning, Info, or Passed checks.

---

## Project Structure

```
dataset-validator-app/
├── app.py               # Flask server — upload, fix, preview, download routes
├── validator_engine.py  # Validation logic (returns structured issue dicts)
├── dataset_validator.py # Original CLI validator (kept as reference)
├── requirements.txt
├── templates/
│   └── index.html       # Single-page HTML shell
└── static/
    ├── css/main.css     # Dark industrial stylesheet
    └── js/app.js        # Frontend logic (upload, render, fix dispatch)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the server

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 3. Using the CLI validator (original)

```bash
python dataset_validator.py path/to/file.csv
python dataset_validator.py data.csv --columns order_id price quantity
python dataset_validator.py data.csv --output report.html
```

---

## Available Fix Actions

| fix_id                   | What it does                                          |
|--------------------------|-------------------------------------------------------|
| `fill_mean`              | Fill NaN with column mean                             |
| `fill_median`            | Fill NaN with column median                           |
| `fill_mode`              | Fill NaN with most frequent value                     |
| `fill_zero`              | Fill NaN with 0                                       |
| `fill_unknown`           | Fill NaN with "Unknown"                               |
| `drop_null_rows`         | Drop rows where column is null                        |
| `drop_full_duplicates`   | Remove fully duplicate rows                           |
| `drop_col_duplicates`    | Remove duplicate rows by a specific column            |
| `cast_to_numeric`        | Strip currency symbols and cast to float              |
| `cast_to_datetime`       | Parse column as datetime                              |
| `cast_to_string`         | Cast column to string                                 |
| `strip_lowercase`        | Strip whitespace and lowercase categorical values     |
| `strip_titlecase`        | Strip whitespace and title-case categorical values    |
| `replace_pseudo_nulls`   | Replace "N/A", "Unknown", "-" etc. with NaN           |
| `drop_negative_rows`     | Remove rows with negative values in a column          |
| `abs_negative`           | Convert negatives to absolute values                  |
| `cap_outliers_iqr`       | Cap extreme outliers to 3×IQR bounds                  |
| `drop_zero_rows`         | Remove rows where column equals zero                  |
| `percent_divide_100`     | Divide column by 100 (percent → fraction)             |
| `drop_column`            | Remove a column entirely                              |
| `drop_empty_columns`     | Remove all fully empty columns                        |
| `drop_full_duplicate_columns` | Remove duplicate column names                    |
| `drop_future_dates`      | Remove rows with future dates in a column             |

---

## GitHub — Creating a Feature Branch

Run these commands from inside your local project directory:

```bash
# 1. Make sure you are on the main branch and up to date
git checkout main
git pull origin main

# 2. Create and switch to a new feature branch
git checkout -b feature/dataset-validator-webapp

# 3. Copy the web app files into your repository
#    (adjust the source path to wherever you saved the files)
cp -r path/to/dataset-validator-app/* .

# 4. Stage all new and modified files
git add .

# 5. Commit
git commit -m "feat: add Dataset Validator web application

- Flask backend with upload, fix, preview, download endpoints
- Refactored validation engine (validator_engine.py) returning
  structured issue dicts with automated fix actions
- Dark industrial frontend (HTML/CSS/JS) with one-click fix buttons
- 9 validation categories, 22 fix action types
- Data preview modal and cleaned CSV download"

# 6. Push the branch to GitHub
git push -u origin feature/dataset-validator-webapp
```

### Open a Pull Request

After pushing, go to your GitHub repository page.
GitHub will show a banner: **"Compare & pull request"** — click it,
add a description, and open the PR for review.

### Merge after testing

```bash
# After the PR is approved, merge locally or via GitHub UI
git checkout main
git merge feature/dataset-validator-webapp
git push origin main

# Clean up the feature branch
git branch -d feature/dataset-validator-webapp
git push origin --delete feature/dataset-validator-webapp
```

---

## Notes

- Sessions are stored **in memory** — restarting the server clears them.
- For production use, replace the in-memory `SESSIONS` dict with a database
  or file-based store (Redis, SQLite, etc.).
- The `UPLOAD_FOLDER` uses the system temp directory; set it to a persistent
  path for production.
