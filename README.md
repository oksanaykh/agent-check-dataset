# 🔍 Dataset Validation Agent

A Python agent for validating CSV datasets before data analysis.
Automatically runs **9 categories of checks** and opens a detailed HTML report in your browser showing every issue found and how to fix it.

---

## 📦 Installation

```bash
pip install pandas numpy
```

---

## 🚀 Usage

### Basic run
```bash
python dataset_validator.py path/to/your/file.csv
```

### With expected columns
```bash
python dataset_validator.py data.csv --columns order_id customer_id product_id price quantity
```

### Save report to a specific file
```bash
python dataset_validator.py data.csv --output report.html
```

> If `--output` is not provided, the report is written to a temp file and opened in your default browser automatically.

### All options
```bash
python dataset_validator.py --help
```

---

## 📋 What the agent checks

| # | Category | What is validated |
|---|----------|-------------------|
| 1 | **Structure** | File loading, empty/duplicate/unnamed/technical columns, dataset size |
| 2 | **Data types** | Correct dtype detection, dates stored as strings, numbers stored as object, mixed types |
| 3 | **Missing values** | Null counts and ratios per column, critical fields, text pseudo-nulls ("N/A", "Unknown", "-") |
| 4 | **Duplicates** | Fully duplicate rows, duplicate IDs, repeated transactions |
| 5 | **Numeric values** | Negative values, suspicious zeros, outliers (3×IQR), discount range, cross-column consistency |
| 6 | **Categorical data** | Case/whitespace duplicates, junk values, high cardinality |
| 7 | **Dates** | Parsing success, date range, future dates, old outliers, ambiguous format, sparse coverage |
| 8 | **Keys & IDs** | Uniqueness, empty IDs, inconsistent format, one ID mapping to multiple categories |
| 9 | **Business logic** | Amount/discount/price consistency, payment method values, product→category integrity |

---

## 📄 Sample report

The HTML report opens automatically in your browser after each run:

- **Summary bar** — counts of Critical / Warnings / Info / Passed checks at a glance
- **Color-coded cards** per issue — 🔴 Critical, 🟡 Warning, 🔵 Info, ✅ OK
- **"How to fix"** hint with a ready-to-use pandas snippet for every issue found

---

## 🔧 Use as a module

```python
from dataset_validator import DatasetValidationAgent

agent = DatasetValidationAgent(
    filepath="my_data.csv",
    expected_columns=["order_id", "customer_id", "product_id", "price", "quantity"]
)

# Run all checks and open the HTML report
agent.run()

# Or save the report to a file
agent.run(output_path="report.html")

# Or run individual checks
agent._load()
agent.check_structure()
agent.check_nulls()
agent.check_duplicates()
print(agent.report.render_html())
```

---

## 🛠 Automatic column type detection

The agent recognises column roles from their names — no configuration needed:

| Role | Keywords matched |
|------|-----------------|
| Date | `date`, `time`, `timestamp`, `created`, `updated`, `period`, `month`, `year` |
| Numeric | `amount`, `price`, `qty`, `quantity`, `total`, `revenue`, `cost`, `discount`, `rate` |
| ID | `_id`, `id`, `identifier`, `code`, `number`, `num`, `key` |
| Category | `category`, `type`, `status`, `region`, `country`, `segment`, `payment`, `method` |
| Critical field | any of the above — null rate >30% triggers a CRITICAL issue |

---

## 📁 Requirements

```
pandas >= 1.3.0
numpy  >= 1.21.0
```

---

## 📝 License

MIT
