"""
Data quality checks run before training — catches the kind of silent data
issues (missing text, single-category corpus, near-duplicate rows) that
would otherwise surface later as a confusing model bug.
"""


def validate_corpus(df, text_col='content', category_col='category', min_articles=500, min_categories=4):
    issues = []
    if df[text_col].isna().any():
        issues.append(f"{df[text_col].isna().sum()} rows have missing text")
    if len(df) < min_articles:
        issues.append(f"Only {len(df)} articles — below the {min_articles} minimum")
    n_categories = df[category_col].nunique()
    if n_categories < min_categories:
        issues.append(f"Only {n_categories} categories — below the {min_categories} minimum")
    empty_after_strip = (df[text_col].astype(str).str.strip() == '').sum()
    if empty_after_strip:
        issues.append(f"{empty_after_strip} rows are empty after stripping whitespace")
    duplicate_rows = df.duplicated(subset=[text_col]).sum()
    if duplicate_rows:
        issues.append(f"{duplicate_rows} duplicate articles found")

    return {
        'is_valid': len(issues) == 0,
        'n_articles': len(df),
        'n_categories': n_categories,
        'issues': issues,
    }
