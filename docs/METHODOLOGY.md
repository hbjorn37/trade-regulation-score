# TRS Methodology

The Trade Regulation Score (TRS) is a daily 0-999 risk index derived from news articles. The thesis used a local LLM to read article text and emit a score, with rule-based keyword weights adjusting the LLM output. This document records that procedure for transparency; reproducing it is optional because the pipeline consumes pre-computed TRS files directly.

## Per-day procedure

1. Collect every article published on date `d` for the target topic.
2. Build three keyword counts over those articles: `risk`, `target country`, `positive`. Each count is the number of articles whose title or body contains at least one keyword from the corresponding list.
3. Concatenate up to 600 characters of article text into a short snippet.
4. Prompt the LLM with the snippet, the keyword counts, and a calibration table mapping integer ranges to qualitative levels.
5. Parse the first integer in `[0, 999]` out of the response. If the LLM fails or refuses, fall back to a 50.
6. Multiply by three weights:
   - `risk_weight`: `1.4` if `risk_ratio > 0.3`, `1.2` for `> 0.2`, `1.1` for `> 0.1`, otherwise `0.8`.
   - `country_weight`: `1.1` for `country_ratio > 0.8`, `1.0` for `> 0.5`, `0.9` for `> 0.2`, otherwise `0.7`.
   - `positive_suppression`: `1.0 - min(1, positive_ratio * 0.8) * 0.9`, floored at `0.1`.
7. Clip to `[10, 999]`.
8. Tag the day with a short rationale label by score band (`baseline`, `low risk`, `elevated risk signals`, `regulation tightening`, `acute spike`).

## Reference implementation

See [`reference/trs_scoring/score_trs.py`](../reference/trs_scoring/score_trs.py). It expects:

- A running local Ollama instance and the model `qwen2.5:14b`.
- A news CSV with columns `date`, `title`, `body`.
- A YAML keyword config (examples for the two case studies are provided in [`reference/trs_scoring/keywords_semi.yaml`](../reference/trs_scoring/keywords_semi.yaml) and [`reference/trs_scoring/keywords_urea.yaml`](../reference/trs_scoring/keywords_urea.yaml)).

The thesis-original keyword lists were Korean. The examples in this repo are English placeholders for documentation purposes; using them on a Korean-language corpus will not reproduce the thesis numbers.

## Why the keyword adjustment exists

In early experiments the LLM alone was sensitive to phrasing changes within the same news event. The keyword counts act as a stabilizing prior: high-risk wording reinforces a high LLM score, positive wording dampens it, and the country mention ratio confirms that the score is attributable to the target country rather than to a similarly worded but unrelated story. This is why both the LLM and the rule-based weights enter the final score.
