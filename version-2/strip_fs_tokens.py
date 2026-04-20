"""
strip_fs_tokens.py
==================
Pre-process M2M100-418M-fs test_results.csv by removing FS marker tokens
(<FS> and -) from word and ID columns, keeping word-ID alignment in sync.

Run on Kaggle:
    !python /kaggle/input/.../strip_fs_tokens.py
    Then point MODEL_RESULTS["M2M100-418M-fs"] to OUTPUT_CSV.

Run locally:
    python strip_fs_tokens.py
    Adjust INPUT_CSV / OUTPUT_CSV paths below.
"""

import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_CSV  = r"e:\nadeen\ssl\final3\m2m-fs-test_results.csv"
OUTPUT_CSV = r"e:\nadeen\ssl\final3\m2m-fs-stripped.csv"

FS_WORDS = {"<FS>", "-"}   # word tokens to strip
# ──────────────────────────────────────────────────────────────────────────────


def _parse(cell) -> list:
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [t.strip() for t in str(cell).split("|") if t.strip()]


def strip_row(words_cell, ids_cell):
    """Remove FS marker positions from both word and ID lists, keeping them aligned."""
    words = _parse(words_cell)
    ids   = _parse(ids_cell)

    # Align: if lengths differ (shouldn't happen on valid data), skip stripping
    if len(words) != len(ids):
        return words_cell, ids_cell

    keep = [i for i, w in enumerate(words) if w not in FS_WORDS]
    clean_words = [words[i] for i in keep]
    clean_ids   = [ids[i]   for i in keep]

    return " | ".join(clean_words), " | ".join(clean_ids)


def main():
    path = Path(INPUT_CSV)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(path)

    # Support both column naming conventions
    if "correct_words" not in df.columns:
        df = df.rename(columns={
            "reference":  "correct_words",
            "prediction": "predicted_words",
            "ref_ids":    "correct_ids",
            "pred_ids":   "predicted_ids",
        })

    before_tokens = 0
    after_tokens  = 0

    out_rows = []
    for _, row in df.iterrows():
        cw, ci = strip_row(row["correct_words"],   row["correct_ids"])
        pw, pi = strip_row(row["predicted_words"], row["predicted_ids"])

        before_tokens += len(_parse(row["correct_words"]))
        after_tokens  += len(_parse(cw))

        new_row = row.copy()
        new_row["correct_words"]   = cw
        new_row["correct_ids"]     = ci
        new_row["predicted_words"] = pw
        new_row["predicted_ids"]   = pi
        out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    stripped = before_tokens - after_tokens
    print(f"  Input : {path.name}  ({len(df)} rows)")
    print(f"  Tokens stripped from correct_words: {stripped} "
          f"({stripped / before_tokens * 100:.1f}% of ref tokens)")
    print(f"  Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
