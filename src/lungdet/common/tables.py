def df_to_md(df):
    """Small markdown table without extra packages (pandas.to_markdown needs 'tabulate')."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    return [head, sep, *("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))]
