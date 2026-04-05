"""
Correlation Analysis: LLM vs Human Annotations vs NPMI
=======================================================
Computes Spearman and Pearson correlations between:
  1. LLM scores vs Human annotations
  2. LLM scores vs NPMI (automated metric)
  3. Human annotations vs NPMI (baseline)

Usage:
  python correlation_analysis.py \
    --results_dir "./results" \
    --tasks_dir "/home/sox45ben/Downloads/LLM Evaluation for DTM/prolific/tasks" \
    --survey_dir "/home/sox45ben/Downloads/LLM Evaluation for DTM/prolific/survey_results" \
    --output    "./correlation_results"
"""

import argparse
import re
import ast
import csv
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

try:
    from scipy.stats import spearmanr, pearsonr
except ImportError:
    os.system("pip install scipy --user")
    from scipy.stats import spearmanr, pearsonr

try:
    import pandas as pd
except ImportError:
    os.system("pip install pandas --user")
    import pandas as pd

DATASET_SURVEY_MAP = {
    "neurips": "NIPS",
    "nyt":     "NYT",
    "unDebates": "UN",
}

DATASET_TASKS_MAP = {
    "neurips": "NIPS",
    "nyt":     "NYT",
    "unDebates": "UN",
}

MODELS = ["llama", "qwen", "mistral", "deepseek"]

DATASET_BASE_YEARS = {
    "neurips":    1987,
    "nyt":        1987,
    "unDebates":  1970,
}
DTM_MODELS = ["DLDA", "DETM"]


def load_human_annotations(survey_dir, dataset):
    keyword = DATASET_SURVEY_MAP.get(dataset, dataset.upper())
    survey_dir = Path(survey_dir)
    survey_file = None
    for f in survey_dir.glob("*.csv"):
        if keyword in f.name:
            survey_file = f
            break
    if survey_file is None:
        print(f"  [Warning] No survey file found for {dataset}")
        return {}
    df = pd.read_csv(survey_file)
    consent_col = [c for c in df.columns if "consent" in c.lower() or "agree" in c.lower()]
    if consent_col:
        df = df[df[consent_col[0]].str.lower().str.contains("agree", na=False)]
    task_ratings = {}
    task_nums = set()
    for col in df.columns:
        if "Word relatedness in Task" in col:
            task_num = int(col.split("Task")[-1].strip())
            task_nums.add(task_num)
    for task_num in sorted(task_nums):
        wr_col = f"Word relatedness in Task {task_num}"
        sm_col = f"Smooth transitions in Task {task_num}"
        if wr_col in df.columns and sm_col in df.columns:
            wr_vals = pd.to_numeric(df[wr_col], errors="coerce").dropna()
            sm_vals = pd.to_numeric(df[sm_col], errors="coerce").dropna()
            task_ratings[task_num] = {
                "word_relatedness": float(wr_vals.mean()) if len(wr_vals) > 0 else None,
                "smoothness":       float(sm_vals.mean()) if len(sm_vals) > 0 else None,
                "n_annotators":     len(wr_vals),
            }
    print(f"  Loaded human annotations: {len(task_ratings)} tasks from {survey_file.name}")
    return task_ratings


def load_task_info(tasks_dir, dtm_model, dataset):
    keyword = DATASET_TASKS_MAP.get(dataset, dataset.upper())
    folder_name = f"{dtm_model}_{keyword}"
    folder = Path(tasks_dir) / folder_name
    task_info_file = None
    if folder.exists():
        for f in folder.glob("*_rating_task_info.csv"):
            task_info_file = f
            break
    if task_info_file is None:
        print(f"  [Warning] No task info file found for {dtm_model}/{dataset}")
        return {}
    df = pd.read_csv(task_info_file)
    task_info = {}
    for _, row in df.iterrows():
        task_no = str(row["task_no"]).strip()
        task_num = int("".join(filter(str.isdigit, task_no.split("_")[-1])))
        task_num = task_num + 1  # DN_0 maps to Task 1 in survey
        try:
            ts = [int(x) for x in re.findall(r"\d+", str(row["time_stamps"]))]
        except Exception:
            ts = []
        task_info[task_num] = {
            "topic_no":    int(row["topic_no"]),
            "time_stamps": list(ts),
        }
    print(f"  Loaded task info: {len(task_info)} tasks from {task_info_file.name}")
    return task_info


def load_npmi_scores(tasks_dir, dtm_model, dataset):
    keyword = DATASET_TASKS_MAP.get(dataset, dataset.upper())
    folder_name = f"{dtm_model}_{keyword}"
    npmi_path = Path(tasks_dir) / folder_name / "tq_c_npmi.csv"
    if not npmi_path.exists():
        print(f"  [Warning] NPMI file not found: {npmi_path}")
        return {}
    df = pd.read_csv(npmi_path)
    npmi_scores = {}
    for _, row in df.iterrows():
        year_idx = int(row.get("year", 0))
        try:
            coh_str = str(row["coherence"])
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", coh_str)]
            for topic_no, val in enumerate(nums):
                npmi_scores[(topic_no, year_idx)] = val
        except Exception:
            pass
    print(f"  Loaded NPMI scores: {len(npmi_scores)} entries from {npmi_path.name}")
    return npmi_scores


def load_llm_scores(results_dir, dtm_model, dataset, model_name):
    coh_path = Path(results_dir) / dtm_model / dataset / f"{model_name}_coherence.csv"
    if not coh_path.exists():
        print(f"  [Warning] LLM results not found: {coh_path}")
        return {}
    df = pd.read_csv(coh_path)
    llm_scores = {}
    for _, row in df.iterrows():
        topic_idx = int(row["topic_idx"])
        year      = int(row["year"])
        tc = row.get("temporal_coherence")
        ts = row.get("temporal_smoothness")
        try:
            tc = float(tc) if tc is not None and str(tc) != "None" else None
            ts = float(ts) if ts is not None and str(ts) != "None" else None
        except Exception:
            tc, ts = None, None
        llm_scores[(topic_idx, year)] = {
            "temporal_coherence":  tc,
            "temporal_smoothness": ts,
        }
    valid = sum(1 for v in llm_scores.values() if v["temporal_coherence"] is not None)
    print(f"  Loaded LLM scores: {len(llm_scores)} entries ({valid} valid) from {coh_path.name}")
    return llm_scores


def safe_corr(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"spearman_r": None, "spearman_p": None,
                "pearson_r":  None, "pearson_p":  None, "n": len(x)}
    # Skip if no variance in either array
    if np.std(x) == 0 or np.std(y) == 0:
        return {"spearman_r": None, "spearman_p": None,
                "pearson_r":  None, "pearson_p":  None, "n": len(x)}
    sp_r, sp_p = spearmanr(x, y)
    pe_r, pe_p = pearsonr(x, y)
    return {
        "spearman_r": round(float(sp_r), 4),
        "spearman_p": round(float(sp_p), 4),
        "pearson_r":  round(float(pe_r), 4),
        "pearson_p":  round(float(pe_p), 4),
        "n":          len(x),
    }


def compute_correlations(results_dir, tasks_dir, survey_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    for dtm_model in DTM_MODELS:
        for dataset in ["neurips", "nyt", "unDebates"]:
            print(f"\n{'='*60}")
            print(f"  {dtm_model} / {dataset}")
            print(f"{'='*60}")

            human     = load_human_annotations(survey_dir, dataset)
            task_info = load_task_info(tasks_dir, dtm_model, dataset)
            npmi      = load_npmi_scores(tasks_dir, dtm_model, dataset)

            if not human or not task_info:
                print(f"  [Skip] Missing human or task info data")
                continue

            for model_name in MODELS:
                print(f"\n  Model: {model_name}")
                llm = load_llm_scores(results_dir, dtm_model, dataset, model_name)
                if not llm:
                    print(f"  [Skip] No LLM results")
                    continue

                llm_coh_list  = []
                llm_sm_list   = []
                human_wr_list = []
                human_sm_list = []
                npmi_list     = []

                for task_num, task_data in task_info.items():
                    topic_idx   = task_data["topic_no"]
                    time_stamps = task_data["time_stamps"]
                    if task_num not in human:
                        continue
                    h_wr = human[task_num]["word_relatedness"]
                    h_sm = human[task_num]["smoothness"]
                    if h_wr is None or h_sm is None:
                        continue

                    llm_coh_vals = []
                    llm_sm_vals  = []
                    npmi_vals    = []

                    base_year = DATASET_BASE_YEARS.get(dataset, 1987)
                    for year_idx in time_stamps:
                        actual_year = base_year + year_idx
                        key_by_year = (topic_idx, actual_year)
                        key_by_idx  = (topic_idx, year_idx)
                        if key_by_year in llm:
                            v = llm[key_by_year]
                            if v["temporal_coherence"] is not None:
                                llm_coh_vals.append(v["temporal_coherence"])
                            if v["temporal_smoothness"] is not None:
                                llm_sm_vals.append(v["temporal_smoothness"])
                        npmi_key = (topic_idx, year_idx)
                        if npmi_key in npmi:
                            npmi_vals.append(npmi[npmi_key])

                    for i, coh_val in enumerate(llm_coh_vals):
                        llm_coh_list.append(coh_val)
                        llm_sm_list.append(llm_sm_vals[i] if i < len(llm_sm_vals) else np.nan)
                        human_wr_list.append(h_wr)
                        human_sm_list.append(h_sm)
                        npmi_list.append(npmi_vals[i] if i < len(npmi_vals) else np.nan)

                if len(llm_coh_list) < 3:
                    print(f"  [Skip] Not enough aligned data points ({len(llm_coh_list)})")
                    continue

                print(f"  Aligned {len(llm_coh_list)} task-level data points")

                r1 = safe_corr(llm_coh_list, human_wr_list)
                r2 = safe_corr(llm_sm_list,  human_sm_list)
                r3 = safe_corr(llm_coh_list, npmi_list)
                r4 = safe_corr(human_wr_list, npmi_list)

                print(f"  LLM_coh vs Human_WR : spearman={r1['spearman_r']} p={r1['spearman_p']}")
                print(f"  LLM_sm  vs Human_SM : spearman={r2['spearman_r']} p={r2['spearman_p']}")
                print(f"  LLM_coh vs NPMI     : spearman={r3['spearman_r']} p={r3['spearman_p']}")
                

                for corr_type, corr_data in [
                    ("LLM_coherence_vs_Human_WR",  r1),
                    ("LLM_smoothness_vs_Human_SM", r2),
                    ("LLM_coherence_vs_NPMI",      r3),
                    
                ]:
                    all_results.append({
                        "dtm_model":  dtm_model,
                        "dataset":    dataset,
                        "llm":        model_name,
                        "comparison": corr_type,
                        "n_tasks":    corr_data["n"],
                        "spearman_r": corr_data["spearman_r"],
                        "spearman_p": corr_data["spearman_p"],
                        "pearson_r":  corr_data["pearson_r"],
                        "pearson_p":  corr_data["pearson_p"],
                        "significant": (corr_data["spearman_p"] or 1) < 0.05,
                    })

    if not all_results:
        print("\n[Warning] No correlation results computed!")
        return

    out_path = Path(output_dir) / "correlation_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[Saved] {len(all_results)} rows -> {out_path}")

    print(f"\n{'='*80}")
    print(f"  CORRELATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'DTM':<8} {'Dataset':<12} {'LLM':<10} {'Comparison':<35} {'Spearman':>10} {'p-val':>8}")
    print(f"{'-'*80}")
    for r in all_results:
        sp_r = f"{r['spearman_r']:.3f}" if r["spearman_r"] is not None else "N/A"
        sp_p = f"{r['spearman_p']:.3f}" if r["spearman_p"] is not None else "N/A"
        sig  = "***" if (r["spearman_p"] or 1) < 0.001 else \
               "**"  if (r["spearman_p"] or 1) < 0.01  else \
               "*"   if (r["spearman_p"] or 1) < 0.05  else ""
        print(f"{r['dtm_model']:<8} {r['dataset']:<12} {r['llm']:<10} {r['comparison']:<35} {sp_r:>10} {sp_p:>8} {sig}")

    json_path = Path(output_dir) / "correlation_summary.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Saved] -> {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--tasks_dir",   required=True)
    parser.add_argument("--survey_dir",  required=True)
    parser.add_argument("--output",      default="./correlation_results")
    args = parser.parse_args()
    compute_correlations(
        results_dir=args.results_dir,
        tasks_dir=args.tasks_dir,
        survey_dir=args.survey_dir,
        output_dir=args.output,
    )

if __name__ == "__main__":
    main()
