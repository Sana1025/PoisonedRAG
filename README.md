# PoisonedRAG Toy Retrieval Experiment

A toy retrieval experiment inspired by the black-box PoisonedRAG setting (`S = Q`). 

This repo measures whether a single poisoned document with `P = Q ⊕ I` is retrieved for:

1. the **exact** target question,
2. **paraphrases** of that question,
3. **unrelated** questions.

---

## Quick start

```bash
git clone https://github.com/Sana1025/PoisonedRAG.git
cd poisonedRAG
pip install sentence-transformers torch matplotlib PyQt5 numpy pypdf

# CLI (prints results to terminal)
python poison_retrieval_experiment.py

# Interactive GUI (4 tabs: summary table, F1 table, per-query top-5, three plots)
python poison_retrieval_gui.py

# Headless plot renderer (writes 3 PNGs without opening a window)
python render_plots_headless.py
```

First run downloads the `sentence-transformers/all-MiniLM-L6-v2` model (~90 MB) from Hugging Face.

---

## What's in the repo
This repository reproduces only the retrieval-stage behavior of PoisonedRAG and does not fine-tune or attack a language model generator.

| File | Purpose |
|------|---------|
| `poison_retrieval_experiment.py` | The experiment. Defines the 51-document corpus, the poisoned passage, the three query groups, runs the retriever, returns a results dict. No UI. |
| `poison_retrieval_gui.py` | PyQt5 viewer. Runs the experiment on a background thread and displays results in four tabs (three tables + three matplotlib plots). |
| `render_plots_headless.py` | Runs the experiment and saves the three plots as PNGs without launching a window. Uses the `Agg` backend. |

---

## Results on the default target

Target question: *"Who won the Nobel Prize in Physics in 2023?"*
Attacker-chosen target answer (false): *"Albert Einstein"*

| Group | n | top-1 | top-5 | top-10 | mean rank | Recall @ k=5 | F1 @ k=5 |
|---|---|---|---|---|---|---|---|
| exact | 1 | 1.00 | 1.00 | 1.00 | 1.0 | 1.00 | 0.333 |
| paraphrase | 5 | 1.00 | 1.00 | 1.00 | 1.0 | 1.00 | 0.333 |
| unrelated | 5 | 0.00 | 0.00 | 0.00 | 38.8 | 0.00 | 0.000 |

The **F1 ceiling of 0.333** is mechanical: with `N=1` poisoned document and `k=5`, precision is capped at `1/5 = 0.20`. The paper achieves F1 0.89–1.00 by injecting `N=5` poisons per target.

### Plots

![top-k hit rate](plot_topk_hit_rate.png)
*Top-k retrieval rate by query group. Exact and paraphrase columns sit at 1.0 across all k; unrelated is empty.*

![precision / recall / F1](plot_precision_recall_f1.png)
*Malicious retrieval precision, recall, and F1 at k=5 per group. Dashed line marks the paper's NQ black-box F1 = 0.96 (Table 1, achieved with N=5).*

![per-query poison rank](plot_poison_rank_per_query.png)
*Rank of the poisoned document for each individual query (log x-axis). Target-related queries pin the poison at rank 1; unrelated queries push it to ranks 14–50.*

---

## Environment

Tested on Windows 11 with:

- Python 3.13.13
- sentence-transformers 5.4.1
- PyTorch 2.10.0 (CPU build)
- matplotlib 3.10.7
- PyQt5 5.15.11
- numpy 2.x
