"""Reference implementation of the TRS scoring procedure used in the thesis.

This script is provided for methodology transparency and is NOT invoked by
``scripts/run.py``. The pipeline consumes the pre-computed ``trs_final_*.csv``
files in ``data/sample/`` directly.

Algorithm (per day):

    1. Collect all news articles for that day for the target topic.
    2. Count three keyword bands (risk, target country, positive).
    3. Send a short prompt to a locally hosted Ollama model (qwen2.5:14b in the
       thesis) asking for an integer score in [0, 999].
    4. Multiplicatively adjust the model score by keyword-derived weights so
       that the final score reflects both the LLM judgement and the rule-based
       signals.

To run it you need:

    - A running Ollama server: ``ollama serve``
    - The qwen2.5:14b model pulled locally: ``ollama pull qwen2.5:14b``
    - A CSV of news with columns: date (YYYY.MM.DD), title, body

Usage::

    uv run python reference/trs_scoring/score_trs.py \\
        --news data/real/semi/news_2017_2022.csv \\
        --output data/real/semi/trs_scored.csv \\
        --keywords reference/trs_scoring/keywords_semi.yaml \\
        --year 2019
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b"


@dataclass
class Keywords:
    risk: list[str]
    target_country: list[str]
    positive: list[str]
    prompt_template: str


def load_keywords(path: Path) -> Keywords:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Keywords(
        risk=raw.get("risk", []),
        target_country=raw.get("target_country", []),
        positive=raw.get("positive", []),
        prompt_template=raw["prompt_template"],
    )


def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    if requests is None:
        raise RuntimeError("The 'requests' package is required: pip install requests")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 20,
            "num_ctx": 4096,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _count_articles_with_any(articles: list[dict], terms: list[str]) -> int:
    if not terms:
        return 0
    return sum(
        1 for a in articles
        if any(t.lower() in (a.get("title", "") + " " + a.get("body", "")).lower() for t in terms)
    )


def score_day(date_str: str, articles: list[dict], keywords: Keywords,
              model: str) -> tuple[float, str, dict]:
    if not articles:
        return 0.0, "no articles", {"article_count": 0}

    risk_signal = _count_articles_with_any(articles, keywords.risk)
    country_mention = _count_articles_with_any(articles, keywords.target_country)
    positive_signal = _count_articles_with_any(articles, keywords.positive)
    n = len(articles)

    combined = " ".join((a.get("title", "") + " " + a.get("body", "")) for a in articles)[:600]
    prompt = keywords.prompt_template.format(
        date=date_str,
        text=combined,
        risk_n=risk_signal,
        country_n=country_mention,
        positive_n=positive_signal,
    )

    response = call_ollama(prompt, model=model)
    digits = re.findall(r"\b(\d{1,3})\b", response)
    valid = [int(d) for d in digits if 0 <= int(d) <= 999]
    base_score = valid[0] if valid else 50

    risk_ratio = risk_signal / n
    country_ratio = country_mention / n
    positive_ratio = positive_signal / n

    risk_weight = 1.4 if risk_ratio > 0.3 else 1.2 if risk_ratio > 0.2 else 1.1 if risk_ratio > 0.1 else 0.8
    country_weight = 1.1 if country_ratio > 0.8 else 1.0 if country_ratio > 0.5 else 0.9 if country_ratio > 0.2 else 0.7
    positive_suppression = max(0.1, 1.0 - min(1.0, positive_ratio * 0.8) * 0.9)

    final_score = max(10.0, min(999.0, base_score * risk_weight * country_weight * positive_suppression))

    if final_score >= 800:
        rationale = "acute spike"
    elif final_score >= 600:
        rationale = "regulation tightening"
    elif final_score >= 400:
        rationale = "elevated risk signals"
    elif final_score >= 200:
        rationale = "low risk"
    else:
        rationale = "baseline"

    details = {
        "article_count": n,
        "country_mention_count": country_mention,
        "risk_signal_count": risk_signal,
        "positive_signal_count": positive_signal,
        "base_score": base_score,
    }
    return final_score, rationale, details


def process_news(news_df: pd.DataFrame, output_file: Path, keywords: Keywords,
                 model: str) -> None:
    processed = set()
    if output_file.exists():
        existing = pd.read_csv(output_file, encoding="utf-8-sig")
        processed = set(existing["date"].tolist())
        print(f"Resuming: {len(processed)} days already scored")

    grouped = news_df.groupby("date")
    print(f"Days to process: {len(grouped)}")

    for date_str, group in grouped:
        if date_str in processed:
            continue
        articles = [
            {"title": str(row.get("title", "")), "body": str(row.get("body", ""))}
            for _, row in group.iterrows()
        ]
        try:
            score, rationale, details = score_day(date_str, articles, keywords, model)
        except Exception as e:
            print(f"  {date_str} -- scoring failed ({e}); using fallback")
            score, rationale, details = 100.0, "fallback", {"article_count": len(articles)}

        row = pd.DataFrame([{
            "date": date_str,
            "trs_score": round(score, 1),
            "rationale": rationale,
            "article_count": details.get("article_count", len(articles)),
            "country_mention_count": details.get("country_mention_count", 0),
            "risk_signal_count": details.get("risk_signal_count", 0),
            "positive_signal_count": details.get("positive_signal_count", 0),
        }])

        mode = "a" if output_file.exists() else "w"
        header = not output_file.exists()
        row.to_csv(output_file, mode=mode, header=header, index=False, encoding="utf-8-sig")
        processed.add(date_str)
        time.sleep(0.5)

    print("Done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--news", required=True, type=Path, help="Path to news CSV (date,title,body)")
    parser.add_argument("--output", required=True, type=Path, help="Output TRS CSV")
    parser.add_argument("--keywords", required=True, type=Path, help="YAML with risk/country/positive lists and prompt_template")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--year", type=int, default=None, help="Process only this year")
    args = parser.parse_args()

    keywords = load_keywords(args.keywords)
    df = pd.read_csv(args.news, encoding="utf-8-sig")
    if args.year is not None:
        df = df[df["date"].astype(str).str.startswith(str(args.year))]
    if df.empty:
        print("No rows match the filter; exiting")
        sys.exit(0)

    process_news(df, args.output, keywords, args.model)


if __name__ == "__main__":
    main()
