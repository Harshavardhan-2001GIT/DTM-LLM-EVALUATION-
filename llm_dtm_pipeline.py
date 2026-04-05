"""
LLM Evaluation Pipeline for Dynamic Topic Models (DLDA / DETM)
===============================================================
Reads topic files in the format:
  <model>/<dataset>/<year>.txt
  where each LINE = one topic, words are space-separated

Example structure:
  DLDA/neurips/1987.txt
  DLDA/nyt/1987.txt
  DLDA/unDebates/1987.txt
  DETM/neurips/1987.txt
  ...

Models evaluated via OpenRouter:
  - meta-llama/llama-3.1-8b-instruct
  - qwen/qwen-2.5-7b-instruct
  - mistralai/Mistral-7B-Instruct-v0.2
  - deepseek/deepseek-chat:free

Tasks:
  1. Temporal Topic Coherence  -> temporal_coherence (1-3), temporal_smoothness (1-3)
  2. Temporal Intrusion Detection -> correct/incorrect intruder detection
  3. Topic Evolution Quality   -> Meaningful evolution / Minor drift / Incoherent change

Usage:
  python llm_dtm_pipeline.py \\
      --data_root /path/to/LLM_Evaluation_for_DTM \\
      --output    ./results \\
      --api_key   sk-or-xxxxxxxx \\
      --model     all \\
      --task      all \\
      --dataset   all          # or: neurips  nyt  unDebates
      --dtm_model all          # or: DLDA  DETM
      --top_n     10           # how many top words per topic to use

Resume support:
  Add --skip_done to skip any output CSV that already exists (safe to re-run).
"""

import argparse
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HF_URL = "https://router.huggingface.co/v1/chat/completions"
HF_KEY = "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"
HF_MODELS = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "mistralai/Mistral-7B-Instruct-v0.2"]  # OpenRouter for llama/qwen

MODELS = {
    "llama":    "meta-llama/llama-3.1-8b-instruct",
    "qwen":     "qwen/qwen-2.5-7b-instruct",
    "mistral":  "mistralai/Mistral-7B-Instruct-v0.2",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

DTM_MODELS = ["DLDA", "DETM"]
DATASETS   = ["neurips", "nyt", "unDebates"]

MAX_RETRIES   = 3
RETRY_DELAY   = 5    # seconds between retries
REQUEST_DELAY = 1.5  # seconds between API calls (rate limit safety)
MAX_TOKENS    = 1024


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_topic_file(filepath, top_n=10):
    """
    Load one year's topic file.
    Each line = one topic; words are space-separated.
    Returns list of topics, each topic = list of top_n words.
    """
    topics = []
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            words = line.strip().split()
            if words:
                topics.append(words[:top_n])
    return topics


def load_dataset(data_root, dtm_model, dataset, top_n=10):
    """
    Load all year files for a given DTM model + dataset.
    Returns: {year(int): [[word, ...], ...]}  (list of topics per year)
    """
    folder = Path(data_root) / dtm_model / dataset
    if not folder.exists():
        print(f"  [Warning] Folder not found: {folder}")
        return {}

    year_topics = {}
    for txt_file in sorted(folder.glob("*.txt")):
        try:
            year = int(txt_file.stem)
        except ValueError:
            continue  # skip non-year files
        year_topics[year] = load_topic_file(txt_file, top_n=top_n)

    if year_topics:
        sample_year = next(iter(year_topics))
        n_topics    = len(year_topics[sample_year])
        print(f"  Loaded {len(year_topics)} years | {n_topics} topics/year "
              f"| from {dtm_model}/{dataset}")
    else:
        print(f"  [Warning] No .txt year files found in {folder}")
    return year_topics


# ─────────────────────────────────────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────────────────────────────────────

def call_openrouter(api_key, model_id, system_prompt, user_prompt, temperature=0.0):
    use_hf = any(m in model_id for m in ["deepseek-ai", "mistralai/Mistral"])
    url = HF_URL if use_hf else OPENROUTER_URL
    key = HF_KEY if use_hf else api_key
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/Harshavardhan-2001GIT/LLM-DTM",
        "X-Title":       "LLM-DTM-Evaluation",
    }
    payload = {
        "model":       model_id,
        "max_tokens":  MAX_TOKENS,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers,
                                 json=payload, timeout=60)
            if resp.status_code == 200:
                msg = resp.json()["choices"][0]["message"]
                text = msg.get("content") or ""
                if not text:
                    text = msg.get("reasoning_content") or ""
                return text.strip()
            elif resp.status_code == 429:
                wait = RETRY_DELAY * attempt
                print(f"\n    [Rate limit] waiting {wait}s ...")
                time.sleep(wait)
            else:
                print(f"\n    [HTTP {resp.status_code}] {resp.text[:200]}")
                time.sleep(RETRY_DELAY)
        except requests.exceptions.Timeout:
            print(f"\n    [Timeout] attempt {attempt}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"\n    [Error] {e}")
            time.sleep(RETRY_DELAY)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PARSE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(text):
    """Try to extract a JSON object from LLM response text."""
    try:
        clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        matches = list(re.finditer(r'\{[^{}]+\}', clean, re.DOTALL))
        for m in reversed(matches):
            try:
                return json.loads(m.group())
            except:
                continue
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — TEMPORAL TOPIC COHERENCE
# ─────────────────────────────────────────────────────────────────────────────

COHERENCE_SYSTEM = (
    "You are an expert in topic modeling and temporal text analysis. "
    "You evaluate topics produced by a Dynamic Topic Model (DTM). "
    "A good topic is semantically coherent and evolves meaningfully over time."
)

def make_coherence_prompt(words_prev, words_curr, words_next):
    def fmt(ws): return "[" + ", ".join(ws) + "]"
    return (
        f"Top words for the same topic at three consecutive time steps:\n\n"
        f"Time t-1: {fmt(words_prev)}\n"
        f"Time t:   {fmt(words_curr)}\n"
        f"Time t+1: {fmt(words_next)}\n\n"
        f"1. Rate word relatedness at time t\n"
        f"   1=not related, 2=somewhat related, 3=very related\n"
        f"2. Rate transition smoothness t-1 -> t -> t+1\n"
        f"   1=not smooth, 2=somewhat smooth, 3=very smooth\n\n"
        f'Reply ONLY in JSON: {{"temporal_coherence": <1|2|3>, "temporal_smoothness": <1|2|3>}}'
    )

def run_coherence(api_key, model_id, year_topics, dtm_model, dataset, llm_name, max_topics=20):
    results = []
    years    = sorted(year_topics.keys())
    n_topics = min(len(year_topics[years[0]]), max_topics)

    for topic_idx in range(n_topics):
        for i in range(1, len(years) - 1):
            y_prev, y_curr, y_next = years[i-1], years[i], years[i+1]
            if (topic_idx >= len(year_topics.get(y_prev, [])) or
                topic_idx >= len(year_topics.get(y_curr, [])) or
                topic_idx >= len(year_topics.get(y_next, []))):
                continue

            prompt = make_coherence_prompt(
                year_topics[y_prev][topic_idx],
                year_topics[y_curr][topic_idx],
                year_topics[y_next][topic_idx],
            )
            print(f"    [Coh] topic={topic_idx:02d} year={y_curr}", end=" ... ", flush=True)
            raw = call_openrouter(api_key, model_id, COHERENCE_SYSTEM, prompt)
            time.sleep(REQUEST_DELAY)

            obj = extract_json(raw) if raw else None
            tc  = int(obj["temporal_coherence"])  if obj and "temporal_coherence"  in obj else None
            ts  = int(obj["temporal_smoothness"]) if obj and "temporal_smoothness" in obj else None
            print(f"coh={tc} smooth={ts}" if tc else f"FAIL: {str(raw)[:60]}")

            results.append({
                "dtm_model":           dtm_model,
                "dataset":             dataset,
                "llm":                 llm_name,
                "topic_idx":           topic_idx,
                "year":                y_curr,
                "temporal_coherence":  tc,
                "temporal_smoothness": ts,
                "raw":                 raw,
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — TEMPORAL INTRUSION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

INTRUSION_SYSTEM = (
    "You are evaluating topic coherence in a Dynamic Topic Model. "
    "One word in the list does NOT belong to the topic. Find it."
)

def sample_intruder(year_topics, year, topic_idx):
    """Pick a random word from a DIFFERENT topic in the same year."""
    topics_this_year = year_topics.get(year, [])
    other_indices    = [i for i in range(len(topics_this_year)) if i != topic_idx]
    if not other_indices:
        return None
    other_topic = topics_this_year[random.choice(other_indices)]
    return random.choice(other_topic) if other_topic else None

def run_intrusion(api_key, model_id, year_topics, dtm_model, dataset, llm_name, max_topics=20):
    results  = []
    years    = sorted(year_topics.keys())
    n_topics = min(len(year_topics[years[0]]), max_topics)

    for topic_idx in range(n_topics):
        for year in years:
            if topic_idx >= len(year_topics.get(year, [])):
                continue
            words = year_topics[year][topic_idx]
            if len(words) < 5:
                continue

            intruder = sample_intruder(year_topics, year, topic_idx)
            if intruder is None:
                continue

            word_list = words[:5] + [intruder]
            random.shuffle(word_list)

            prompt = (
                f"Year: {year}\n"
                f"Topic words: {word_list}\n\n"
                f"Which word is LEAST related to the others? "
                f"Reply with ONE word only."
            )
            print(f"    [Int] topic={topic_idx:02d} year={year} intruder={intruder}", end=" ... ", flush=True)
            raw      = call_openrouter(api_key, model_id, INTRUSION_SYSTEM, prompt, temperature=0.0)
            time.sleep(REQUEST_DELAY)
            detected = raw.strip().lower().split()[0] if raw else None
            correct  = (detected == intruder.lower()) if detected else False
            print(f"detected={detected} correct={correct}")

            results.append({
                "dtm_model": dtm_model,
                "dataset":   dataset,
                "llm":       llm_name,
                "topic_idx": topic_idx,
                "year":      year,
                "intruder":  intruder,
                "detected":  detected,
                "correct":   correct,
                "raw":       raw,
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — TOPIC EVOLUTION QUALITY
# ─────────────────────────────────────────────────────────────────────────────

EVOLUTION_SYSTEM = (
    "You are evaluating how a topic evolves over time in a Dynamic Topic Model. "
    "A good evolution reflects a real-world semantic shift rather than random noise."
)

EVOLUTION_LABELS = ["meaningful evolution", "minor drift", "incoherent change"]

def make_evolution_prompt(topic_idx, window_years, year_topics):
    lines = "\n".join(
        f"Year {y}: {year_topics[y][topic_idx]}"
        for y in window_years
        if topic_idx < len(year_topics.get(y, []))
    )
    return (
        f"Topic evolution over time:\n\n{lines}\n\n"
        f"Does this represent a meaningful semantic change?\n"
        f"Choose ONE: Meaningful evolution | Minor drift | Incoherent change\n"
        f"Then give ONE sentence justification.\n\n"
        f'Reply ONLY in JSON: {{"label": "...", "justification": "..."}}'
    )

def parse_evolution(raw):
    obj = extract_json(raw) if raw else None
    if obj and "label" in obj:
        label = obj["label"].strip().lower()
        for lbl in EVOLUTION_LABELS:
            if label.startswith(lbl[:8]):
                return {"label": obj["label"], "justification": obj.get("justification", "")}
    if raw:
        for lbl in EVOLUTION_LABELS:
            if lbl in raw.lower():
                return {"label": lbl.title(), "justification": raw[:200]}
    return None

def run_evolution(api_key, model_id, year_topics, dtm_model, dataset, llm_name, window=3, max_topics=20):
    results  = []
    years    = sorted(year_topics.keys())
    n_topics = min(len(year_topics[years[0]]), max_topics)

    for topic_idx in range(n_topics):
        # non-overlapping windows of `window` years
        windows = [years[i:i+window] for i in range(0, len(years) - window + 1, window)]
        if not windows:
            windows = [years]

        for win in windows:
            prompt    = make_evolution_prompt(topic_idx, win, year_topics)
            label_str = f"{win[0]}-{win[-1]}"
            print(f"    [Evo] topic={topic_idx:02d} years={label_str}", end=" ... ", flush=True)
            raw    = call_openrouter(api_key, model_id, EVOLUTION_SYSTEM, prompt)
            time.sleep(REQUEST_DELAY)
            parsed = parse_evolution(raw)
            print(parsed["label"] if parsed else f"FAIL: {str(raw)[:60]}")

            results.append({
                "dtm_model":     dtm_model,
                "dataset":       dataset,
                "llm":           llm_name,
                "topic_idx":     topic_idx,
                "year_start":    win[0],
                "year_end":      win[-1],
                "label":         parsed["label"]         if parsed else None,
                "justification": parsed["justification"] if parsed else None,
                "raw":           raw,
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(rows, path):
    if not rows:
        print(f"    [Skip] no rows -> {path}")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"    [Saved] {len(rows)} rows -> {path}")


def already_done(path):
    """Return True if output CSV already exists with at least one data row."""
    p = Path(path)
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        lines = f.readlines()
    return len(lines) > 1


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Pipeline for DLDA/DETM topic models"
    )
    parser.add_argument("--data_root", required=True,
        help="Root folder containing DLDA/ and DETM/ subfolders "
             "(e.g. /home/sox45ben/Downloads/LLM_Evaluation_for_DTM/prolific)")
    parser.add_argument("--output",    required=True,
        help="Output directory for result CSVs")
    parser.add_argument("--api_key",   required=True,
        help="OpenRouter API key (sk-or-...)")
    parser.add_argument("--model",     default="all",
        choices=list(MODELS.keys()) + ["all"],
        help="LLM to use (default: all)")
    parser.add_argument("--task",      default="all",
        choices=["coherence", "intrusion", "evolution", "all"],
        help="Evaluation task to run (default: all)")
    parser.add_argument("--dtm_model", default="all",
        choices=DTM_MODELS + ["all"],
        help="DTM model folder (default: all)")
    parser.add_argument("--dataset",   default="all",
        choices=DATASETS + ["all"],
        help="Dataset folder (default: all)")
    parser.add_argument("--top_n",     type=int, default=10,
        help="Top N words per topic to use (default: 10)")
    parser.add_argument("--window",    type=int, default=3,
        help="Year window size for evolution task (default: 3)")
    parser.add_argument("--max_topics", type=int, default=20, help="Max number of topics to evaluate (default: 20)")
    parser.add_argument("--seed",      type=int, default=42,
        help="Random seed for intrusion shuffling (default: 42)")
    parser.add_argument("--skip_done", action="store_true",
        help="Skip combinations whose output CSV already exists (resume mode)")
    args = parser.parse_args()

    random.seed(args.seed)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    dtm_models = DTM_MODELS if args.dtm_model == "all" else [args.dtm_model]
    datasets   = DATASETS   if args.dataset   == "all" else [args.dataset]
    llm_models = MODELS     if args.model     == "all" else {args.model: MODELS[args.model]}

    print(f"\n{'='*65}")
    print(f"  LLM-DTM Evaluation Pipeline")
    print(f"  data_root : {args.data_root}")
    print(f"  DTM models: {dtm_models}")
    print(f"  Datasets  : {datasets}")
    print(f"  LLMs      : {list(llm_models.keys())}")
    print(f"  Tasks     : {args.task}")
    print(f"  Top-N     : {args.top_n} words per topic")
    print(f"{'='*65}\n")

    for dtm in dtm_models:
        for ds in datasets:
            print(f"\n{'─'*65}")
            print(f"  Loading: {dtm} / {ds}")
            year_topics = load_dataset(args.data_root, dtm, ds, top_n=args.top_n)
            if not year_topics:
                print(f"  [Skip] No data for {dtm}/{ds}")
                continue

            for llm_name, model_id in llm_models.items():
                print(f"\n  LLM: {llm_name}  |  {model_id}")

                # Task 1: Coherence
                if args.task in ("coherence", "all"):
                    out = str(out_root / dtm / ds / f"{llm_name}_coherence.csv")
                    if args.skip_done and already_done(out):
                        print(f"  [Skip] already done: {out}")
                    else:
                        print(f"\n  [Task 1] Temporal Topic Coherence")
                        rows = run_coherence(args.api_key, model_id,
                                             year_topics, dtm, ds, llm_name, max_topics=args.max_topics)
                        save_csv(rows, out)

                # Task 2: Intrusion
                if args.task in ("intrusion", "all"):
                    out = str(out_root / dtm / ds / f"{llm_name}_intrusion.csv")
                    if args.skip_done and already_done(out):
                        print(f"  [Skip] already done: {out}")
                    else:
                        print(f"\n  [Task 2] Temporal Intrusion Detection")
                        rows = run_intrusion(args.api_key, model_id,
                                             year_topics, dtm, ds, llm_name, max_topics=args.max_topics)
                        save_csv(rows, out)

                # Task 3: Evolution
                if args.task in ("evolution", "all"):
                    out = str(out_root / dtm / ds / f"{llm_name}_evolution.csv")
                    if args.skip_done and already_done(out):
                        print(f"  [Skip] already done: {out}")
                    else:
                        print(f"\n  [Task 3] Topic Evolution Quality")
                        rows = run_evolution(args.api_key, model_id,
                                             year_topics, dtm, ds, llm_name, max_topics=args.max_topics,
                                             window=args.window)
                        save_csv(rows, out)

    print(f"\n{'='*65}")
    print(f"  Done!  Results saved to: {out_root}/")
    print(f"\n  Output structure:")
    print(f"    results/DLDA/neurips/llama_coherence.csv")
    print(f"    results/DLDA/neurips/llama_intrusion.csv")
    print(f"    results/DLDA/neurips/llama_evolution.csv")
    print(f"    results/DETM/nyt/qwen_coherence.csv  ... etc.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
