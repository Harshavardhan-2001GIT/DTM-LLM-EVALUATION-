# DTM-LLM-EVALUATION-
# Automated Evaluation of Dynamic Topic Models Using Large Language Models

> A Comparative Study with Human Judgment and Temporal Metrics  
> RPTU Kaiserslautern |Supervisor: Charu Karakkaparambil James

---

## Overview

This project investigates whether Large Language Models (LLMs) can automatically evaluate **Dynamic Topic Models (DTMs)** in a way that correlates with human judgment and outperforms traditional automated metrics like NPMI.

Dynamic Topic Models capture how topics evolve over time in large text corpora. Evaluating them is challenging because:
- Topics change across time slices, requiring assessment of both coherence and temporal consistency
- Human evaluation is expensive and hard to scale
- Traditional metrics (NPMI, TTS, TTC) may not fully reflect interpretability

This project fills the gap by systematically evaluating LLM-based assessment of DTMs for the first time.

---

## Models Evaluated

### DTM Models
| Model | Description |
|-------|-------------|
| DLDA  | Dynamic Latent Dirichlet Allocation |
| DETM  | Dynamic Embedded Topic Model |

### Datasets
| Dataset | Description | Years |
|---------|-------------|-------|
| NeurIPS | Neural Information Processing Systems papers | 1987–2019 |
| NYT | New York Times news articles | Multi-year |
| unDebates | UN General Assembly Debates | Multi-year |

### LLMs Used
| Model | Size | Provider | API |
|-------|------|----------|-----|
| Llama 3.1 | 8B | Meta | OpenRouter |
| Qwen 2.5 | 7B | Alibaba | OpenRouter |
| Mistral | 7B | Mistral AI | HuggingFace |
| DeepSeek R1 Distill | 8B | DeepSeek | HuggingFace |

---

## Evaluation Tasks

### Task 1 — Temporal Topic Coherence
For each topic at time t, the LLM is shown top words at t-1, t, and t+1 and asked to rate:
- **Word relatedness** at time t (1=not related, 2=somewhat, 3=very related)
- **Transition smoothness** from t-1 → t → t+1 (1=not smooth, 2=somewhat, 3=very smooth)

### Task 2 — Temporal Intrusion Detection
An intruder word from a different topic is inserted into the topic word list. The LLM must identify the odd word out. Measures topic coherence at a single time step.

### Task 3 — Topic Evolution Quality
The LLM evaluates topic word evolution across multiple time windows and classifies it as:
- **Meaningful evolution** — real-world semantic shift
- **Minor drift** — small gradual change
- **Incoherent change** — noisy or random shift

---

## Project Structure

```
LLM-DTM/
│
├── llm_dtm_pipeline.py          # Main evaluation pipeline
├── correlation_analysis.py      # LLM vs Human vs NPMI correlation
├── README.md                    # This file
│
├── results/                     # All LLM evaluation outputs
│   ├── DLDA/
│   │   ├── neurips/
│   │   │   ├── llama_coherence.csv
│   │   │   ├── llama_intrusion.csv
│   │   │   ├── llama_evolution.csv
│   │   │   ├── qwen_coherence.csv
│   │   │   ├── mistral_coherence.csv
│   │   │   └── deepseek_coherence.csv
│   │   ├── nyt/
│   │   └── unDebates/
│   └── DETM/
│       ├── neurips/
│       ├── nyt/
│       └── unDebates/
│
└── correlation_results/         # Correlation analysis outputs
    ├── correlation_results.csv
    └── correlation_summary.json
```

---

## Setup & Usage

### Requirements
```bash
pip install requests scipy pandas numpy
```

### API Keys Required
- **OpenRouter** (Llama, Qwen): Get free key at [openrouter.ai](https://openrouter.ai)
- **HuggingFace** (Mistral, DeepSeek): Get free key at [huggingface.co](https://huggingface.co/settings/tokens)

### Run Evaluation Pipeline

```bash
# Test with one model first
python llm_dtm_pipeline.py \
  --data_root /path/to/prolific \
  --output ./results \
  --api_key YOUR_OPENROUTER_KEY \
  --model llama \
  --dtm_model DLDA \
  --dataset neurips \
  --task coherence

# Full run — all models, all datasets, all tasks
python llm_dtm_pipeline.py \
  --data_root /path/to/prolific \
  --output ./results \
  --api_key YOUR_API_KEY \
  --model all \
  --dtm_model all \
  --dataset all \
  --task all \
  --top_n 10 \
  --max_topics 20 \
  --skip_done
```

### Pipeline Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | required | Path to folder containing DLDA/ and DETM/ |
| `--output` | required | Output directory for result CSVs |
| `--api_key` | required | OpenRouter or HuggingFace API key |
| `--model` | all | LLM to use: llama, qwen, mistral, deepseek, all |
| `--dtm_model` | all | DTM model: DLDA, DETM, all |
| `--dataset` | all | Dataset: neurips, nyt, unDebates, all |
| `--task` | all | Task: coherence, intrusion, evolution, all |
| `--top_n` | 10 | Top N words per topic |
| `--max_topics` | 20 | Max topics to evaluate per dataset |
| `--skip_done` | False | Skip already completed CSVs (resume mode) |

### Run Correlation Analysis

```bash
python correlation_analysis.py \
  --results_dir ./results \
  --tasks_dir /path/to/prolific/tasks \
  --survey_dir /path/to/prolific/survey_results \
  --output ./correlation_results
```

---

## Output Format

### Coherence Results (llama_coherence.csv)
| Column | Description |
|--------|-------------|
| dtm_model | DLDA or DETM |
| dataset | neurips, nyt, unDebates |
| llm | Model name |
| topic_idx | Topic index (0-49) |
| year | Year of evaluation |
| temporal_coherence | Score 1-3 |
| temporal_smoothness | Score 1-3 |

### Intrusion Results (llama_intrusion.csv)
| Column | Description |
|--------|-------------|
| topic_idx | Topic index |
| year | Year |
| intruder | Planted intruder word |
| detected | Word identified by LLM |
| correct | True/False |

### Evolution Results (llama_evolution.csv)
| Column | Description |
|--------|-------------|
| topic_idx | Topic index |
| year_start | Window start year |
| year_end | Window end year |
| label | Meaningful evolution / Minor drift / Incoherent change |
| justification | One sentence explanation |

---

## Correlation Results

Spearman and Pearson correlations computed between:
- **LLM coherence vs Human word relatedness** (main research question)
- **LLM smoothness vs Human smoothness** (secondary research question)
- **LLM coherence vs NPMI** (LLM vs automated metric)
- **Human vs NPMI** (baseline)

---

## References

1. Charu Karakkaparambil James et al. *Evaluating Dynamic Topic Models.* ACL 2024.
2. Stammbach et al. *Revisiting Automated Topic Model Evaluation with LLMs.* EMNLP 2023.
3. Yang et al. *LLM Reading Tea Leaves: Automatically Evaluating Topic Models with LLMs.* TACL 2025.

---

## Citation

```bibtex
@project{llm_dtm_2025,
  title     = {Automated Evaluation of Dynamic Topic Models Using Large Language Models},
  author    = {Harshavardhan},
  year      = {2025},
  institution = {RPTU Kaiserslautern},
  supervisor  = {Charu Karakkaparambil James}
}
```
