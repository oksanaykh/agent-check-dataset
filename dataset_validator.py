"""
Dataset Validation Agent
========================
Проверяет CSV-датасет перед анализом данных по 9 категориям проверок.
Возвращает подробный отчёт о найденных проблемах и способах их устранения.

Использование:
    python dataset_validator.py path/to/your/file.csv

    # Указать ожидаемые колонки (опционально):
    python dataset_validator.py data.csv --columns order_id customer_id product_id price quantity

    # Сохранить отчёт в файл:
    python dataset_validator.py data.csv --output report.txt
"""

import sys
import argparse
import re
from pathlib import Path
from datetime import datetime
from io import StringIO

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Установите зависимости: pip install pandas numpy")
    sys.exit(1)


# ─────────────────────────────────────────────
#  Константы
# ─────────────────────────────────────────────

CRITICAL_FIELDS_KEYWORDS = [
    "date", "дата", "time", "время",
    "amount", "сумма", "total", "итого", "revenue",
    "price", "цена", "стоимость",
    "quantity", "qty", "количество",
    "customer", "клиент", "client",
    "product", "товар", "продукт",
    "order", "заказ",
    "category", "категория",
    "region", "регион",
    "id",
]

DATE_KEYWORDS = ["date", "дата", "time", "время", "period", "month", "year"]
ID_KEYWORDS = ["id", "_id", "identifier", "code", "код", "num", "number"]
NUMERIC_KEYWORDS = ["amount", "price", "qty", "quantity", "total", "sum", "count",
                    "revenue", "cost", "discount", "rate", "percent", "value"]
CATEGORY_KEYWORDS = ["category", "type", "status", "method", "region", "country",
                     "city", "segment", "channel", "gender", "department",
                     "категория", "тип", "статус", "регион", "страна"]

PSEUDO_NULL_VALUES = {"n/a", "na", "nan", "null", "none", "nil", "-", "--", "---",
                      "unknown", "неизвестно", "не указано", "нет данных",
                      "missing", "empty", "", " "}

TECHNICAL_COLUMN_PATTERNS = [
    r"^unnamed",
    r"^index$",
    r"^\.index$",
    r"^col\d+$",
    r"^field\d+$",
    r"^column\d+$",
    r"^_c\d+$",
]

SEVERITY = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵", "OK": "✅"}


# ─────────────────────────────────────────────
#  Вспомогательные функции
# ─────────────────────────────────────────────

def _matches_any(name: str, keywords: list[str]) -> bool:
    n = name.lower()
    return any(kw in n for kw in keywords)


def _is_id_column(name: str) -> bool:
    return _matches_any(name, ID_KEYWORDS)


def _is_date_column(name: str, series: pd.Series) -> bool:
    if _matches_any(name, DATE_KEYWORDS):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    return False


def _is_numeric_column(name: str, series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    return _matches_any(name, NUMERIC_KEYWORDS)


def _is_category_column(name: str, series: pd.Series) -> bool:
    if _matches_any(name, CATEGORY_KEYWORDS):
        return True
    if series.dtype == object:
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


def _try_parse_dates(series: pd.Series):
    """Возвращает (parsed_series, success_ratio, detected_format)"""
    common_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d.%m.%Y", "%Y/%m/%d", "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
    ]
    sample = series.dropna().astype(str).head(200)
    best_parsed = None
    best_ratio = 0.0
    best_fmt = None
    for fmt in common_formats:
        try:
            parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
            ratio = parsed.notna().mean()
            if ratio > best_ratio:
                best_ratio = ratio
                best_parsed = parsed
                best_fmt = fmt
        except Exception:
            continue
    # fallback: infer
    try:
        parsed_inf = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
        ratio_inf = parsed_inf.notna().mean()
        if ratio_inf > best_ratio:
            best_ratio = ratio_inf
            best_parsed = parsed_inf
            best_fmt = "auto-inferred"
    except Exception:
        pass
    return best_parsed, best_ratio, best_fmt


# ─────────────────────────────────────────────
#  Класс отчёта
# ─────────────────────────────────────────────

class ValidationReport:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.sections: list[dict] = []
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

    def render(self) -> str:
        lines = []
        sep = "=" * 70

        lines.append(sep)
        lines.append("  ОТЧЁТ ВАЛИДАЦИИ ДАТАСЕТА")
        lines.append(sep)
        lines.append(f"  Файл:    {self.filepath}")
        lines.append(f"  Время:   {self.timestamp}")
        lines.append(sep)

        # Итог
        c = self.summary_counts
        lines.append("\n📊 ИТОГОВАЯ СВОДКА")
        lines.append("-" * 40)
        lines.append(f"  {SEVERITY['CRITICAL']} КРИТИЧЕСКИХ проблем : {c.get('CRITICAL', 0)}")
        lines.append(f"  {SEVERITY['WARNING']} ПРЕДУПРЕЖДЕНИЙ       : {c.get('WARNING', 0)}")
        lines.append(f"  {SEVERITY['INFO']} ИНФОРМАЦИОННЫХ       : {c.get('INFO', 0)}")
        lines.append(f"  {SEVERITY['OK']} ВСЁ ХОРОШО           : {c.get('OK', 0)}")

        if c.get("CRITICAL", 0) == 0 and c.get("WARNING", 0) == 0:
            lines.append("\n✅ Датасет прошёл валидацию без серьёзных замечаний.")
        elif c.get("CRITICAL", 0) > 0:
            lines.append("\n⚠️  Обнаружены КРИТИЧЕСКИЕ проблемы. Анализ без исправлений может дать некорректные результаты.")
        else:
            lines.append("\n⚠️  Обнаружены предупреждения. Рекомендуется исправить перед анализом.")

        # Группировка по секциям
        from collections import defaultdict
        by_section = defaultdict(list)
        for item in self.sections:
            by_section[item["section"]].append(item)

        for section, items in by_section.items():
            lines.append(f"\n{'─' * 70}")
            lines.append(f"  {section}")
            lines.append("─" * 70)
            for item in items:
                icon = SEVERITY.get(item["severity"], "•")
                lines.append(f"\n  {icon} [{item['severity']}] {item['message']}")
                if item["details"]:
                    for detail_line in item["details"].strip().split("\n"):
                        lines.append(f"      {detail_line}")
                if item["fix"]:
                    lines.append(f"    💡 КАК ИСПРАВИТЬ: {item['fix']}")

        lines.append(f"\n{'=' * 70}")
        lines.append("  КОНЕЦ ОТЧЁТА")
        lines.append("=" * 70)
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Агент валидации
# ─────────────────────────────────────────────

class DatasetValidationAgent:

    def __init__(self, filepath: str, expected_columns: list[str] | None = None):
        self.filepath = filepath
        self.expected_columns = expected_columns or []
        self.df: pd.DataFrame | None = None
        self.report = ValidationReport(filepath)

    # ── 1. Загрузка ──────────────────────────────────────────────────────────

    def _load(self) -> bool:
        encodings = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]
        separators = [",", ";", "\t", "|"]
        path = Path(self.filepath)

        if not path.exists():
            self.report.add_issue(
                "1. СТРУКТУРА ДАТАСЕТА", "CRITICAL",
                f"Файл не найден: {self.filepath}",
                fix="Проверьте путь к файлу."
            )
            return False

        if path.suffix.lower() != ".csv":
            self.report.add_issue(
                "1. СТРУКТУРА ДАТАСЕТА", "WARNING",
                f"Расширение файла не .csv: {path.suffix}",
                fix="Убедитесь, что файл является CSV-файлом."
            )

        for enc in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(self.filepath, encoding=enc, sep=sep,
                                     low_memory=False)
                    if df.shape[1] > 1 or sep == ",":
                        self.df = df
                        self.report.add_issue(
                            "1. СТРУКТУРА ДАТАСЕТА", "INFO",
                            f"Файл загружен. Кодировка: {enc}, разделитель: '{sep}'",
                        )
                        return True
                except Exception:
                    continue

        self.report.add_issue(
            "1. СТРУКТУРА ДАТАСЕТА", "CRITICAL",
            "Не удалось загрузить файл ни с одной из стандартных кодировок и разделителей.",
            fix="Проверьте кодировку файла (UTF-8 рекомендуется) и формат разделителя."
        )
        return False

    # ── 1. Структура ─────────────────────────────────────────────────────────

    def check_structure(self):
        df = self.df
        section = "1. СТРУКТУРА ДАТАСЕТА"

        # Размер
        self.report.add_issue(section, "INFO",
                               f"Размер датасета: {df.shape[0]:,} строк × {df.shape[1]} колонок")

        # Пустой датасет
        if df.empty:
            self.report.add_issue(section, "CRITICAL", "Датасет пустой (0 строк).",
                                  fix="Проверьте источник данных.")
            return

        if df.shape[0] < 5:
            self.report.add_issue(section, "WARNING",
                                  f"Очень мало строк: {df.shape[0]}. Анализ может быть ненадёжным.")

        # Заголовки
        unnamed = [c for c in df.columns if str(c).strip() == ""]
        if unnamed:
            self.report.add_issue(section, "CRITICAL",
                                  f"Найдены колонки без названия: {len(unnamed)} шт.",
                                  fix="Добавьте заголовки или удалите лишние колонки из CSV.")

        # Полностью пустые колонки
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            self.report.add_issue(section, "WARNING",
                                  f"Полностью пустые колонки: {empty_cols}",
                                  fix="Удалите колонки: df.drop(columns=[...], inplace=True)")

        # Дублирующиеся названия
        dupes = [c for c in df.columns if list(df.columns).count(c) > 1]
        dupes = list(set(dupes))
        if dupes:
            self.report.add_issue(section, "CRITICAL",
                                  f"Дублирующиеся названия колонок: {dupes}",
                                  fix="Переименуйте колонки: df.columns = [...]")

        # Технические колонки
        tech_cols = []
        for col in df.columns:
            for pat in TECHNICAL_COLUMN_PATTERNS:
                if re.match(pat, str(col).lower()):
                    tech_cols.append(col)
                    break
        if tech_cols:
            self.report.add_issue(section, "WARNING",
                                  f"Технические/служебные колонки без аналитической ценности: {tech_cols}",
                                  fix="Удалите: df.drop(columns=[...], inplace=True)")

        # Ожидаемые колонки
        if self.expected_columns:
            missing = [c for c in self.expected_columns if c not in df.columns]
            if missing:
                self.report.add_issue(section, "CRITICAL",
                                      f"Ожидаемые колонки отсутствуют: {missing}",
                                      fix="Проверьте источник данных или уточните список колонок.")
            else:
                self.report.add_issue(section, "OK",
                                      "Все ожидаемые колонки присутствуют.")

        if not empty_cols and not dupes and not unnamed and not tech_cols:
            self.report.add_issue(section, "OK", "Структура датасета выглядит корректно.")

    # ── 2. Типы данных ────────────────────────────────────────────────────────

    def check_types(self):
        df = self.df
        section = "2. ТИПЫ ДАННЫХ"

        type_summary = []
        problems = []

        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            sample_vals = series.dropna().head(5).tolist()

            # Дата читается как строка
            if series.dtype == object and _matches_any(col, DATE_KEYWORDS):
                parsed, ratio, fmt = _try_parse_dates(series)
                if ratio > 0.7:
                    problems.append((col, "WARNING",
                                     f"Колонка '{col}' выглядит как дата, но тип — object (строка). "
                                     f"Формат определён: {fmt}, успешно: {ratio:.0%}",
                                     f"df['{col}'] = pd.to_datetime(df['{col}'], format='{fmt}', errors='coerce')"))
                else:
                    problems.append((col, "CRITICAL",
                                     f"Колонка '{col}' — вероятно дата, но парсинг не удался (< 70%). "
                                     f"Примеры: {sample_vals[:3]}",
                                     f"Проверьте формат даты и очистите значения. "
                                     f"Попробуйте: pd.to_datetime(df['{col}'], infer_datetime_format=True, errors='coerce')"))

            # Число читается как строка
            elif series.dtype == object and _matches_any(col, NUMERIC_KEYWORDS):
                clean = series.dropna().astype(str).str.replace(r"[%,$€£₽\s]", "", regex=True).str.replace(",", ".")
                converted = pd.to_numeric(clean, errors="coerce")
                ratio = converted.notna().mean()
                if ratio > 0.7:
                    problems.append((col, "WARNING",
                                     f"Колонка '{col}' — числовая, но тип object. "
                                     f"Примеры: {sample_vals[:3]}. Конвертация успешна на {ratio:.0%}",
                                     f"Очистите и конвертируйте: "
                                     f"df['{col}'] = pd.to_numeric(df['{col}'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')"))

            # Смешанные типы
            if series.dtype == object:
                sample_clean = series.dropna().head(500)
                num_ratio = pd.to_numeric(sample_clean, errors="coerce").notna().mean()
                if 0.1 < num_ratio < 0.9:
                    problems.append((col, "WARNING",
                                     f"Колонка '{col}' содержит смешанные типы (~{num_ratio:.0%} числовых). "
                                     f"Примеры: {sample_vals[:5]}",
                                     "Разделите или очистите колонку. Числовые и нечисловые значения не должны смешиваться."))

            type_summary.append(f"  {col:<35} {dtype}")

        self.report.add_issue(section, "INFO",
                               "Типы колонок:\n" + "\n".join(type_summary))

        for col, sev, msg, fix in problems:
            self.report.add_issue(section, sev, msg, fix=fix)

        if not problems:
            self.report.add_issue(section, "OK",
                                   "Типы данных определены корректно без явных проблем.")

    # ── 3. Пропуски ───────────────────────────────────────────────────────────

    def check_nulls(self):
        df = self.df
        section = "3. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ"
        total = len(df)

        null_counts = df.isnull().sum()
        has_nulls = null_counts[null_counts > 0]

        if has_nulls.empty:
            # Псевдо-пропуски
            pass
        else:
            details_lines = []
            for col, cnt in has_nulls.items():
                flag = " ← КРИТИЧНОЕ ПОЛЕ" if _is_critical_field(str(col)) else ""
                details_lines.append(f"  {col:<35} {cnt:>6} ({_pct(cnt, total)}){flag}")
            self.report.add_issue(section, "WARNING",
                                   f"Найдены пропуски в {len(has_nulls)} колонках:",
                                   fix="Решите стратегию: удаление строк (dropna), заполнение медианой/модой/0 (fillna) или оставить как есть с пометкой.",
                                   details="\n".join(details_lines))

        # Высокий % пропусков в критичных полях
        for col in df.columns:
            if _is_critical_field(str(col)):
                ratio = df[col].isnull().mean()
                if ratio > 0.3:
                    self.report.add_issue(section, "CRITICAL",
                                           f"КРИТИЧНОЕ ПОЛЕ '{col}' содержит {ratio:.0%} пропусков.",
                                           fix=f"Восстановите данные из источника или удалите строки: df.dropna(subset=['{col}'], inplace=True)")

        # Строки с множеством пропусков
        row_null_ratio = df.isnull().mean(axis=1)
        mostly_empty = (row_null_ratio > 0.5).sum()
        if mostly_empty > 0:
            self.report.add_issue(section, "WARNING",
                                   f"Строк, где >50% полей пустые: {mostly_empty} ({_pct(mostly_empty, total)})",
                                   fix="Рассмотрите удаление таких строк: df = df[df.isnull().mean(axis=1) <= 0.5]")

        # Псевдо-пропуски (текстовые)
        pseudo_found = {}
        for col in df.select_dtypes(include="object").columns:
            vals = df[col].dropna().astype(str).str.strip().str.lower()
            pseudo_cnt = vals.isin(PSEUDO_NULL_VALUES).sum()
            if pseudo_cnt > 0:
                pseudo_found[col] = pseudo_cnt
        if pseudo_found:
            details = "\n".join(f"  {c:<35} {n}" for c, n in pseudo_found.items())
            self.report.add_issue(section, "WARNING",
                                   "Текстовые псевдо-пропуски ('N/A', 'Unknown', '-', 'null' и т.д.):",
                                   fix="Замените: df.replace(['N/A','Unknown','-','null'], np.nan, inplace=True)",
                                   details=details)

        if has_nulls.empty and not pseudo_found:
            self.report.add_issue(section, "OK", "Пропущенные значения не обнаружены.")

    # ── 4. Дубликаты ──────────────────────────────────────────────────────────

    def check_duplicates(self):
        df = self.df
        section = "4. ДУБЛИКАТЫ"

        # Полные дубликаты
        full_dupes = df.duplicated().sum()
        if full_dupes > 0:
            self.report.add_issue(section, "CRITICAL",
                                   f"Полных дублирующихся строк: {full_dupes} ({_pct(full_dupes, len(df))})",
                                   fix="Удалите: df.drop_duplicates(inplace=True)")
        else:
            self.report.add_issue(section, "OK", "Полных дубликатов строк не найдено.")

        # Дубликаты по ID-колонкам
        id_cols = [c for c in df.columns if _is_id_column(str(c))]
        for col in id_cols:
            dupes = df[col].dropna().duplicated().sum()
            if dupes > 0:
                pct = _pct(dupes, df[col].notna().sum())
                # Для transaction_id — это критично
                sev = "CRITICAL" if any(kw in str(col).lower() for kw in ["transaction", "order", "заказ"]) else "WARNING"
                self.report.add_issue(section, sev,
                                       f"Дублирующиеся значения в '{col}': {dupes} ({pct})",
                                       fix=f"Проверьте: df[df['{col}'].duplicated(keep=False)].sort_values('{col}')")
            else:
                self.report.add_issue(section, "OK", f"Колонка '{col}': ID уникальны.")

        # Мягкие дубликаты (одинаковые клиент + товар + дата + сумма)
        date_cols = [c for c in df.columns if _matches_any(str(c), DATE_KEYWORDS)]
        amount_cols = [c for c in df.columns if _matches_any(str(c), ["amount", "total", "price", "сумма"])]
        customer_cols = [c for c in df.columns if _matches_any(str(c), ["customer", "клиент", "client"])]
        product_cols = [c for c in df.columns if _matches_any(str(c), ["product", "товар", "product_id"])]

        soft_key = customer_cols[:1] + product_cols[:1] + date_cols[:1] + amount_cols[:1]
        soft_key = [c for c in soft_key if c in df.columns]
        if len(soft_key) >= 3:
            soft_dupes = df.duplicated(subset=soft_key).sum()
            if soft_dupes > 0:
                self.report.add_issue(section, "WARNING",
                                       f"Вероятные дубли транзакций (совпадают {soft_key}): {soft_dupes}",
                                       fix=f"Проверьте: df[df.duplicated(subset={soft_key}, keep=False)]")

    # ── 5. Числовые значения ──────────────────────────────────────────────────

    def check_numerics(self):
        df = self.df
        section = "5. ЧИСЛОВЫЕ ЗНАЧЕНИЯ"
        found_issues = False

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in num_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            col_lower = str(col).lower()

            # Отрицательные значения
            neg_fields = ["quantity", "qty", "price", "amount", "total", "revenue",
                          "cost", "количество", "цена", "сумма"]
            if _matches_any(col, neg_fields):
                neg_cnt = (series < 0).sum()
                if neg_cnt > 0:
                    found_issues = True
                    self.report.add_issue(section, "CRITICAL",
                                           f"'{col}': {neg_cnt} отрицательных значений (мин: {series.min():.2f})",
                                           fix=f"Удалите или исправьте: df = df[df['{col}'] >= 0]")

            # Нулевые значения в критичных полях
            zero_suspect = ["price", "amount", "total", "цена", "сумма"]
            if _matches_any(col, zero_suspect):
                zero_cnt = (series == 0).sum()
                if zero_cnt > 0:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                           f"'{col}': {zero_cnt} нулевых значений",
                                           fix=f"Проверьте: df[df['{col}'] == 0] — это ошибка или реальные данные?")

            # Процентные поля — диапазон 0–100 или 0–1
            pct_fields = ["discount", "rate", "percent", "скидка", "ставка"]
            if _matches_any(col, pct_fields):
                out_of_range = ((series < 0) | (series > 100)).sum()
                if out_of_range > 0:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                           f"'{col}': {out_of_range} значений вне диапазона [0, 100]",
                                           fix="Проверьте единицы измерения (проценты vs доли). Возможно, нужно умножить на 100 или разделить.")

            # Выбросы (IQR-метод)
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)).sum()
                if outliers > 0 and outliers / len(series) > 0.01:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                           f"'{col}': {outliers} экстремальных выбросов (3×IQR). "
                                           f"Диапазон: [{series.min():.2f}, {series.max():.2f}], медиана: {series.median():.2f}",
                                           fix="Проверьте выбросы: df[df['{col}'] > q3 + 3*iqr]. Это ошибка данных или реальные события?")

        # Логическая согласованность: TotalAmount ≈ Qty * Price * (1 - Discount)
        qty_col = next((c for c in df.columns if _matches_any(str(c), ["quantity", "qty", "количество"])), None)
        price_col = next((c for c in df.columns if _matches_any(str(c), ["price", "цена"])), None)
        total_col = next((c for c in df.columns if _matches_any(str(c), ["total", "amount", "сумма", "итого"])), None)

        if qty_col and price_col and total_col:
            try:
                disc_col = next((c for c in df.columns if _matches_any(str(c), ["discount", "скидка"])), None)
                subset = df[[qty_col, price_col, total_col]].dropna()
                if disc_col and disc_col in df.columns:
                    subset = df[[qty_col, price_col, total_col, disc_col]].dropna()
                    disc = subset[disc_col]
                    if disc.max() > 1:
                        disc = disc / 100
                    expected = subset[qty_col] * subset[price_col] * (1 - disc)
                else:
                    expected = subset[qty_col] * subset[price_col]
                actual = subset[total_col]
                tol = 0.05
                mismatch = ((actual - expected).abs() / (expected.abs() + 1e-9) > tol).sum()
                if mismatch > 0:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                           f"Логическая несогласованность: {mismatch} строк, где {total_col} ≠ {qty_col}×{price_col}"
                                           + (f"×(1-{disc_col})" if disc_col else "") + " (отклонение >5%)",
                                           fix="Проверьте формулу расчёта суммы. Возможно, применяются скидки, налоги или округления.")
            except Exception:
                pass

        if not found_issues:
            self.report.add_issue(section, "OK", "Числовые значения в допустимых диапазонах.")

    # ── 6. Категориальные данные ──────────────────────────────────────────────

    def check_categoricals(self):
        df = self.df
        section = "6. КАТЕГОРИАЛЬНЫЕ ДАННЫЕ"
        found_issues = False

        cat_cols = [c for c in df.columns if _is_category_column(str(c), df[c])]

        for col in cat_cols:
            series = df[col].dropna().astype(str)
            unique_vals = series.unique()
            nunique = len(unique_vals)

            # Регистровые дубли
            lower_vals = series.str.strip().str.lower()
            lower_unique = lower_vals.nunique()
            if lower_unique < nunique:
                dupes = nunique - lower_unique
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                       f"'{col}': {dupes} дублей из-за различия регистра или пробелов",
                                       fix=f"df['{col}'] = df['{col}'].str.strip().str.lower()")

            # Мусорные категории
            pseudo_cats = lower_vals[lower_vals.isin(PSEUDO_NULL_VALUES)]
            if len(pseudo_cats) > 0:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                       f"'{col}': {len(pseudo_cats)} мусорных значений ('Unknown', '-' и т.д.)",
                                       fix=f"df['{col}'].replace([...], np.nan, inplace=True)")

            # Слишком высокая кардинальность
            if nunique > 100:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                       f"'{col}': высокая кардинальность — {nunique} уникальных значений. "
                                       "Может быть проблемой для визуализаций и группировок.",
                                       fix="Проверьте: это действительно категория или ID/свободный текст?")
            elif nunique <= 30:
                vals_str = ", ".join(sorted(str(v) for v in unique_vals[:20]))
                suffix = f" ... (+{nunique - 20} ещё)" if nunique > 20 else ""
                self.report.add_issue(section, "INFO",
                                       f"'{col}' ({nunique} значений): {vals_str}{suffix}")

        if not cat_cols:
            self.report.add_issue(section, "INFO",
                                   "Категориальные колонки не обнаружены автоматически.")
        elif not found_issues:
            self.report.add_issue(section, "OK",
                                   "Категориальные данные выглядят чистыми.")

    # ── 7. Даты ───────────────────────────────────────────────────────────────

    def check_dates(self):
        df = self.df
        section = "7. ДАТЫ И ВРЕМЕННОЙ ДИАПАЗОН"
        now = datetime.now()
        found_issues = False

        date_cols = [c for c in df.columns
                     if pd.api.types.is_datetime64_any_dtype(df[c]) or _matches_any(str(c), DATE_KEYWORDS)]

        for col in date_cols:
            series = df[col]

            # Парсинг, если строка
            if series.dtype == object:
                parsed, ratio, fmt = _try_parse_dates(series)
                if ratio < 0.8:
                    found_issues = True
                    self.report.add_issue(section, "CRITICAL",
                                           f"'{col}': только {ratio:.0%} значений успешно парсятся как дата.",
                                           fix=f"Проверьте формат. Пример: pd.to_datetime(df['{col}'], dayfirst=True, errors='coerce')")
                    continue
                else:
                    series = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
                    if ratio < 1.0:
                        failed = series.isnull().sum() - df[col].isnull().sum()
                        if failed > 0:
                            found_issues = True
                            self.report.add_issue(section, "WARNING",
                                                   f"'{col}': {failed} значений не удалось распарсить как дату.",
                                                   fix="Проверьте нестандартные форматы в этих строках.")

            valid_dates = series.dropna()
            if len(valid_dates) == 0:
                continue

            min_date = valid_dates.min()
            max_date = valid_dates.max()
            self.report.add_issue(section, "INFO",
                                   f"'{col}': диапазон {min_date.date()} → {max_date.date()} "
                                   f"({(max_date - min_date).days} дней)")

            # Даты из будущего
            future = (valid_dates > pd.Timestamp(now)).sum()
            if future > 0:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                       f"'{col}': {future} дат из будущего.",
                                       fix="Проверьте корректность данных. Если это ошибки — удалите или исправьте.")

            # Слишком старые даты
            very_old = (valid_dates < pd.Timestamp("1990-01-01")).sum()
            if very_old > 0:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                       f"'{col}': {very_old} дат до 1990 года. Проверьте на ошибки.",
                                       fix="Проверьте: df[df['{col}'] < '1990-01-01']")

            # Неоднозначный формат
            sample_str = df[col].dropna().astype(str).head(10).tolist()
            ambiguous = [v for v in sample_str if re.match(r"^\d{2}/\d{2}/\d{4}$", str(v))]
            if ambiguous:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                       f"'{col}': неоднозначный формат дат (dd/mm/yyyy vs mm/dd/yyyy). "
                                       f"Примеры: {ambiguous[:3]}",
                                       fix=f"Явно укажите параметр dayfirst: pd.to_datetime(df['{col}'], dayfirst=True)")

            # Пробелы во временном ряду (если выглядит как транзакционные данные)
            if len(valid_dates) > 30:
                days_series = valid_dates.dt.date
                all_days = pd.date_range(min_date, max_date, freq="D")
                missing_days = len(all_days) - days_series.nunique()
                coverage = days_series.nunique() / len(all_days)
                if coverage < 0.5:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                           f"'{col}': низкая плотность дат — {days_series.nunique()} уникальных дней "
                                           f"из {len(all_days)} возможных ({coverage:.0%}). "
                                           "Возможны пробелы в данных или неполная загрузка.",
                                           fix="Проверьте, все ли периоды загружены из источника.")

        if not date_cols:
            self.report.add_issue(section, "INFO",
                                   "Колонки с датами не обнаружены. Если дата есть — проверьте тип.")
        elif not found_issues:
            self.report.add_issue(section, "OK", "Проблем с датами не обнаружено.")

    # ── 8. Ключи и идентификаторы ─────────────────────────────────────────────

    def check_keys(self):
        df = self.df
        section = "8. КЛЮЧИ И ИДЕНТИФИКАТОРЫ"
        found_issues = False

        id_cols = [c for c in df.columns if _is_id_column(str(c))]

        for col in id_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            # Пустые ID
            null_cnt = df[col].isnull().sum()
            if null_cnt > 0:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                       f"'{col}': {null_cnt} пустых ID",
                                       fix=f"Строки без ID могут быть ошибками: df[df['{col}'].isnull()]")

            # Непоследовательный формат ID
            str_series = series.astype(str)
            lengths = str_series.str.len().value_counts()
            if len(lengths) > 3:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                       f"'{col}': непоследовательная длина ID — {dict(lengths.head(5))}",
                                       fix="Проверьте на ошибки форматирования или смешанные источники данных.")

            # Разные типы в одной колонке (числа и строки)
            try:
                numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
                if 0.05 < numeric_ratio < 0.95:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                           f"'{col}': смешанные типы ID ({numeric_ratio:.0%} числовых)",
                                           fix="Приведите к единому типу: df['{col}'] = df['{col}'].astype(str)")
            except Exception:
                pass

            # Один CustomerID → разные категории (нарушение однозначности)
            cat_cols = [c for c in df.columns
                        if _is_category_column(str(c), df[c]) and c != col]
            for cat_col in cat_cols[:2]:
                mapping = df.groupby(col)[cat_col].nunique()
                inconsistent = (mapping > 1).sum()
                if inconsistent > 0:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                           f"'{col}': {inconsistent} ID связаны с несколькими значениями '{cat_col}'. "
                                           "Возможна непоследовательность в данных.",
                                           fix=f"Проверьте: df.groupby('{col}')['{cat_col}'].nunique().sort_values(ascending=False).head(10)")

        if not id_cols:
            self.report.add_issue(section, "INFO",
                                   "Колонки-идентификаторы не обнаружены автоматически.")
        elif not found_issues:
            self.report.add_issue(section, "OK", "Ключи и идентификаторы выглядят корректно.")

    # ── 9. Бизнес-логика ─────────────────────────────────────────────────────

    def check_business_logic(self):
        df = self.df
        section = "9. БИЗНЕС-ЛОГИКА И СОГЛАСОВАННОСТЬ"
        found_issues = False

        # Скидка не должна увеличивать сумму
        disc_col = next((c for c in df.columns if _matches_any(str(c), ["discount", "скидка"])), None)
        total_col = next((c for c in df.columns if _matches_any(str(c), ["total", "amount", "сумма"])), None)
        price_col = next((c for c in df.columns if _matches_any(str(c), ["price", "цена"])), None)

        if disc_col and total_col and price_col:
            try:
                subset = df[[disc_col, total_col, price_col]].dropna()
                disc_num = pd.to_numeric(subset[disc_col], errors="coerce")
                total_num = pd.to_numeric(subset[total_col], errors="coerce")
                price_num = pd.to_numeric(subset[price_col], errors="coerce")
                # Сумма с ненулевой скидкой не должна превышать цену
                bad = ((disc_num > 0) & (total_num > price_num)).sum()
                if bad > 0:
                    found_issues = True
                    self.report.add_issue(section, "WARNING",
                                           f"{bad} строк: сумма ({total_col}) > цены ({price_col}) при ненулевой скидке — подозрительно.",
                                           fix="Проверьте логику расчёта итоговой суммы.")
            except Exception:
                pass

        # Метод оплаты — допустимые значения
        payment_col = next((c for c in df.columns
                            if _matches_any(str(c), ["payment", "оплата", "pay_method", "payment_method"])), None)
        if payment_col:
            unique_payments = df[payment_col].dropna().astype(str).str.strip().str.lower().unique()
            suspicious = [v for v in unique_payments
                          if v in PSEUDO_NULL_VALUES or len(v) > 50]
            if suspicious:
                found_issues = True
                self.report.add_issue(section, "WARNING",
                                       f"'{payment_col}': подозрительные значения способа оплаты: {suspicious[:5]}",
                                       fix="Стандартизируйте значения (Credit Card, Cash, Online и т.д.)")

        # Категория товара — ProductID не должен относиться к разным категориям
        product_id_col = next((c for c in df.columns
                               if "product" in str(c).lower() and "id" in str(c).lower()), None)
        cat_col = next((c for c in df.columns if _matches_any(str(c), ["category", "категория"])), None)

        if product_id_col and cat_col:
            try:
                mapping = df.groupby(product_id_col)[cat_col].nunique()
                multi_cat = (mapping > 1).sum()
                if multi_cat > 0:
                    found_issues = True
                    self.report.add_issue(section, "CRITICAL",
                                           f"{multi_cat} product_id относятся к нескольким категориям — нарушение целостности.",
                                           fix=f"Проверьте: df.groupby('{product_id_col}')['{cat_col}'].nunique().sort_values(ascending=False).head()")
            except Exception:
                pass

        if not found_issues:
            self.report.add_issue(section, "OK",
                                   "Логическая согласованность данных не нарушена.")

    # ── Запуск всех проверок ──────────────────────────────────────────────────

    def run(self) -> str:
        print("🔍 Запуск валидации датасета...")

        if not self._load():
            return self.report.render()

        print("  [1/9] Структура...")
        self.check_structure()
        print("  [2/9] Типы данных...")
        self.check_types()
        print("  [3/9] Пропущенные значения...")
        self.check_nulls()
        print("  [4/9] Дубликаты...")
        self.check_duplicates()
        print("  [5/9] Числовые значения...")
        self.check_numerics()
        print("  [6/9] Категориальные данные...")
        self.check_categoricals()
        print("  [7/9] Даты...")
        self.check_dates()
        print("  [8/9] Ключи и идентификаторы...")
        self.check_keys()
        print("  [9/9] Бизнес-логика...")
        self.check_business_logic()

        print("✅ Валидация завершена.\n")
        return self.report.render()


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Validation Agent — проверяет CSV перед анализом данных",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python dataset_validator.py data.csv
  python dataset_validator.py data.csv --columns order_id customer_id price quantity
  python dataset_validator.py data.csv --output validation_report.txt
        """
    )
    parser.add_argument("filepath", help="Путь к CSV-файлу")
    parser.add_argument(
        "--columns", nargs="*", default=[],
        help="Ожидаемые названия колонок (опционально)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Сохранить отчёт в файл (опционально)"
    )
    args = parser.parse_args()

    agent = DatasetValidationAgent(
        filepath=args.filepath,
        expected_columns=args.columns
    )
    report_text = agent.run()
    print(report_text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n📄 Отчёт сохранён в: {args.output}")


if __name__ == "__main__":
    main()
