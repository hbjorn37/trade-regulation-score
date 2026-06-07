# Reference: TRS Scoring with Ollama

The scripts in this directory produced the Trade Regulation Score (TRS) values used in the thesis. They are not part of the reproducibility pipeline in [`scripts/run.py`](../../scripts/run.py). The pipeline consumes pre-computed `trs_final_*.csv` files from [`data/sample/`](../../data/sample/) directly.

## Why this is separate

TRS scoring runs an open-source LLM over a corpus of news articles. The thesis used:

- News source: Naver News (Korean), Baidu News (Chinese) for the urea case
- LLM: `qwen2.5:14b` served locally via Ollama
- Prompt: a structured request asking for an integer in [0, 999] with category guidance, augmented by keyword counts as side information

We split this out for three reasons.

1. It requires a separately installed and running Ollama server plus a GPU (or substantial CPU and patience) to score several thousand days.
2. The keyword lists are language- and domain-specific. The example configs in [`keywords_semi.yaml`](keywords_semi.yaml) and [`keywords_urea.yaml`](keywords_urea.yaml) are English placeholders; the thesis used the Korean originals.
3. The output of scoring is the same shape as the synthetic `trs_final_*.csv` files shipped under [`data/sample/`](../../data/sample/), so the pipeline does not need to be aware of where the scores came from.

## How to run

```bash
# 1. Install and start Ollama, pull the model
ollama serve
ollama pull qwen2.5:14b

# 2. Place a news corpus at data/real/<dataset>/news_2017_2022.csv with
#    columns date (YYYY.MM.DD), title, body.

# 3. Score one year at a time so you can resume on interruption
uv run python reference/trs_scoring/score_trs.py \
    --news data/real/semi/news_2017_2022.csv \
    --output data/real/semi/trs_scored.csv \
    --keywords reference/trs_scoring/keywords_semi.yaml \
    --year 2019

# 4. Concatenate per-year outputs into a single trs_final_2017_2022.csv and
#    drop it next to the import data under data/real/<dataset>/.
```

The script writes one row per date; rerunning it skips dates already present in the output file, which makes long runs resumable.
