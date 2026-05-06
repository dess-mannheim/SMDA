"""
Labeling pipeline for the Reddit causal inference teaching exercise.

Reads r/fatpeoplehate, r/TumblrInAction, r/KotakuInAction, and r/MensRights
JSONL comment dumps, classifies each comment with GroNLP/hateBERT, aggregates
to a user×month panel, and saves two .parquet.zst files for use in the notebook.

Output
------
  reddit_users.parquet.zst   — one row per user, pre-treatment covariates
  reddit_panel.parquet.zst   — one row per (user, month), outcome variables

Analysis design
---------------
  Treated  : users active in r/fatpeoplehate Jan–May 2015 who also appear in
             at least one control subreddit (TIA / KIA / MensRights).
  Control  : users active in the control subreddits who never appeared in fph.
  Outcome  : hate_rate and monthly_comments in the control subreddits,
             measured Jan–Oct 2015 (June skipped — half-month at ban).

Usage
-----
  pip install transformers torch tqdm zstandard pandas pyarrow
  python label_reddit_comments.py

Runtime
-------
  CPU-only : ~30–60 min for 600 users.
  GPU      : ~5–10 min.
"""

import io
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import zstandard as zstd
from tqdm import tqdm
from transformers import pipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent

FPH_FILE   = DATA_DIR / "r_fatpeoplehate_comments.jsonl"
TIA_FILE   = DATA_DIR / "r_TumblrInAction_comments.jsonl"
KIA_FILE   = DATA_DIR / "r_KotakuInAction_comments.jsonl"
MR_FILE    = DATA_DIR / "r_MensRights_comments.jsonl"

OUT_USERS = DATA_DIR / "reddit_users.parquet.zst"
OUT_PANEL = DATA_DIR / "reddit_panel.parquet.zst"

SKIP_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", ""}

MIN_PRE_COMMENTS = 10   # minimum pre-period comments to include a user
N_TREATED        = 300  # most-active treated users to keep
N_CONTROL        = 300  # control users (randomly sampled from top 1500)
BATCH_SIZE       = 64   # hateBERT inference batch size

# Calendar: ban was June 10, 2015; we skip June entirely.
PRE_START  = pd.Timestamp("2015-01-01")
PRE_END    = pd.Timestamp("2015-06-01")   # exclusive
BAN_MONTH  = pd.Timestamp("2015-06-01")
POST_START = pd.Timestamp("2015-07-01")
POST_END   = pd.Timestamp("2015-11-01")   # exclusive

PRE_MONTHS  = pd.date_range("2015-01-01", "2015-05-01", freq="MS")
POST_MONTHS = pd.date_range("2015-07-01", "2015-10-01", freq="MS")
ALL_MONTHS  = list(PRE_MONTHS) + list(POST_MONTHS)

PANEL_TS_START = PRE_START.timestamp()
PANEL_TS_END   = POST_END.timestamp()
PRE_TS_END     = PRE_END.timestamp()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_month(ts_unix: int) -> pd.Timestamp:
    return pd.Timestamp(ts_unix, unit="s").normalize().replace(day=1)


def save_zst(df: pd.DataFrame, path: Path) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    compressed = zstd.ZstdCompressor(level=9).compress(buf.getvalue())
    path.write_bytes(compressed)
    mb = path.stat().st_size / 1e6
    print(f"  Saved {path.name}  ({len(df):,} rows, {mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Phase 1: Identify treated users from fph
# ---------------------------------------------------------------------------
def load_fph_pre_authors() -> set:
    print("Phase 1: Loading r/fatpeoplehate pre-ban authors (Jan–May 2015)...")
    authors = set()
    with open(FPH_FILE) as f:
        for line in f:
            d = json.loads(line)
            if d["author"] in SKIP_AUTHORS:
                continue
            if d["created_utc"] < PRE_START.timestamp():
                continue
            if d["created_utc"] >= PRE_TS_END:
                continue
            authors.add(d["author"])
    print(f"  {len(authors):,} unique fph pre-ban authors")
    return authors


# ---------------------------------------------------------------------------
# Phase 2: Load control subreddit comments
# ---------------------------------------------------------------------------
def load_control_comments(fph_authors: set) -> pd.DataFrame:
    print("\nPhase 2: Loading control subreddit comments (Jan–Oct 2015)...")
    files = {
        "TumblrInAction": TIA_FILE,
        "KotakuInAction": KIA_FILE,
        "MensRights":     MR_FILE,
    }
    rows = []
    for sub, path in files.items():
        n = 0
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                ts = d["created_utc"]
                if ts < PANEL_TS_START or ts >= PANEL_TS_END:
                    continue
                author = d["author"]
                if author in SKIP_AUTHORS:
                    continue
                rows.append({
                    "author":    author,
                    "subreddit": sub,
                    "month":     to_month(ts),
                    "score":     d.get("score", 0),
                    "body":      d.get("body", ""),
                    "treated":   int(author in fph_authors),
                })
                n += 1
        print(f"  r/{sub}: {n:,} comments loaded")

    df = pd.DataFrame(rows)
    print(f"  Total: {len(df):,} comments, "
          f"{df.author.nunique():,} authors")
    return df


# ---------------------------------------------------------------------------
# Phase 3: Select users
# ---------------------------------------------------------------------------
def select_users(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    print("\nPhase 3: Selecting users...")
    pre = df[df["month"] < BAN_MONTH]
    counts = pre.groupby(["author", "treated"]).size().reset_index(name="n_pre")
    counts = counts[counts["n_pre"] >= MIN_PRE_COMMENTS]

    treated_pool = (counts[counts["treated"] == 1]
                    .sort_values("n_pre", ascending=False)
                    .head(N_TREATED))

    control_pool = counts[counts["treated"] == 0].sort_values("n_pre", ascending=False)
    top_control = control_pool.head(1500)
    control_sample = top_control.sample(n=min(N_CONTROL, len(top_control)),
                                        random_state=42)

    selected = pd.concat([treated_pool, control_sample])["author"]
    print(f"  Treated: {len(treated_pool)}, Control: {len(control_sample)}")
    return df[df["author"].isin(selected)].copy()


# ---------------------------------------------------------------------------
# Phase 4: Classify with hateBERT
# ---------------------------------------------------------------------------
def run_hatebert(df: pd.DataFrame) -> pd.DataFrame:
    print("\nPhase 4: Running GroNLP/hateBERT...")
    classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-hate-latest",
        truncation=True,
        max_length=512,
        device='mps',   # CPU; set to 0 for first GPU
    )

    texts = df["body"].tolist()
    labels = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  batches"):
        batch = texts[i : i + BATCH_SIZE]
        results = classifier(batch, batch_size=BATCH_SIZE)
        labels.extend(int(r["label"] == "HATE") for r in results)

    df = df.copy()
    df["hate"] = labels
    print(f"  Mean hate rate: {np.mean(labels):.3f}")
    return df


# ---------------------------------------------------------------------------
# Phase 5: Aggregate to user×month panel
# ---------------------------------------------------------------------------
def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    print("\nPhase 5: Aggregating to user×month panel...")
    # Expand to full month grid for every user
    user_meta = df[["author", "treated"]].drop_duplicates()
    grid = user_meta.assign(key=1).merge(
        pd.DataFrame({"month": ALL_MONTHS, "key": 1}), on="key"
    ).drop(columns="key")

    agg = (df.groupby(["author", "month"])
             .agg(monthly_comments=("body", "count"),
                  hate_rate=("hate", "mean"))
             .reset_index())

    panel = grid.merge(agg, on=["author", "month"], how="left")
    panel["monthly_comments"] = panel["monthly_comments"].fillna(0).astype(int)
    panel["hate_rate"] = panel["hate_rate"].fillna(0.0)
    panel["post"] = (panel["month"] >= BAN_MONTH).astype(int)
    panel = panel.rename(columns={"author": "user_id"})
    print(f"  {len(panel):,} user-month observations")
    return panel


# ---------------------------------------------------------------------------
# Phase 6: Compute user-level pre-period covariates
# ---------------------------------------------------------------------------
def build_users(df: pd.DataFrame) -> pd.DataFrame:
    print("\nPhase 6: Computing pre-period covariates...")
    pre = df[df["month"] < BAN_MONTH].copy()

    sub_counts = (pre.groupby(["author", "subreddit"])["body"]
                  .count().reset_index(name="n"))
    totals = sub_counts.groupby("author")["n"].sum().rename("pre_total_comments")
    diversity = sub_counts.groupby("author")["subreddit"].nunique().rename("pre_subreddit_diversity")
    max_share = (sub_counts.groupby("author")
                 .apply(lambda g: g["n"].max() / g["n"].sum(), include_groups=False)
                 .rename("pre_max_sub_share"))

    user_agg = (pre.groupby("author")
                .agg(treated=("treated", "first"),
                     pre_hate_rate=("hate", "mean"),
                     pre_score=("score", "mean"))
                .reset_index())
    user_agg = (user_agg
                .merge(totals.reset_index(), on="author")
                .merge(diversity.reset_index(), on="author")
                .merge(max_share.reset_index(), on="author"))
    user_agg["pre_avg_monthly_activity"] = user_agg["pre_total_comments"] / len(PRE_MONTHS)
    user_agg = user_agg.rename(columns={"author": "user_id"})
    print(f"  {len(user_agg)} users  "
          f"({user_agg.treated.sum()} treated, {(1-user_agg.treated).sum()} control)")
    return user_agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)

    fph_authors = load_fph_pre_authors()
    comments    = load_control_comments(fph_authors)
    comments    = select_users(comments, rng)
    comments    = run_hatebert(comments)
    panel       = build_panel(comments)
    users       = build_users(comments)

    print("\nSaving...")
    save_zst(users, OUT_USERS)
    save_zst(panel, OUT_PANEL)
    print("\nDone.")


if __name__ == "__main__":
    main()
