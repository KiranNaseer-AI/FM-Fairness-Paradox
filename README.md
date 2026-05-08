# When More Parameters Hurt: Foundation Model Priors Amplify Worst-Client Disparity Under Extreme Federated Heterogeneity

[![Paper](https://img.shields.io/badge/Paper-FL@FM--IJCAI%202026-blue)](https://arxiv.org/abs/ARXIV_ID_HERE)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

**Kiran Naseer, Umar Shoaib**  
University of Gujrat, Pakistan  
*Submitted to FL@FM Workshop, IJCAI 2026*

---

## TL;DR

> Foundation models are widely assumed to be universally beneficial in federated learning. We show this assumption **fails precisely where it matters most**. Under extreme data heterogeneity (α=0.1), DistilBERT+LoRA produces a **worst-client accuracy gap of 50.1%** — 56% larger than a simple TextCNN — despite having 25× more parameters. We call this the **FM Fairness Paradox**.

---

## The FM Fairness Paradox

| Heterogeneity | TextCNN Gap | DistilBERT+LoRA Gap | Winner (Fairness) |
|:---:|:---:|:---:|:---:|
| α = 0.1 (Extreme) | 32.2% | **50.1%** ❌ | TextCNN |
| α = 0.5 (High) | 3.7% | **1.3%** ✅ | DistilBERT+LoRA |
| α = 1.0 (Medium) | 5.6% | **0.0%** ✅ | DistilBERT+LoRA |
| α = 5.0 (Near-IID) | 3.1% | **0.0%** ✅ | DistilBERT+LoRA |

**Key finding:** There is a critical heterogeneity threshold at α ≈ 0.5. Below it, deploying a foundation model makes minority clients *worse off*. Above it, the FM is the fairer model.

---

## Abstract

Federated learning (FL) is increasingly used to fine-tune foundation models (FMs) on distributed private data. The community largely assumes that large-scale pretraining serves as a *rising tide that lifts all boats* in federated settings. However, our experiments reveal that these powerful priors can hinder rather than help the most disadvantaged clients under extreme heterogeneity.

Through controlled experiments on federated text classification, we compare worst-client accuracy between TextCNN (2.7M parameters) and DistilBERT with Low-Rank Adaptation (LoRA, 66M parameters) across four Non-IID heterogeneity levels. Under extreme label skew (α=0.1), DistilBERT+LoRA produces a worst-client accuracy gap of 50.1% — 56% larger than TextCNN's 32.2% gap, despite having 25× more parameters and extensive pretraining. Under moderate heterogeneity (α ≥ 0.5), the pattern reverses: the FM nearly eliminates the gap.

We further show that an inverse-weighted LoRA aggregation method (FedAvgW) does not resolve the disparity, suggesting aggregation reweighting alone is insufficient. Our results highlight the need for mechanisms that explicitly protect minority clients before deploying foundation models in high-stakes federated contexts such as healthcare and education.

---

## Key Contributions

1. **First controlled empirical comparison** of worst-client robustness between a lightweight model and a foundation model with PEFT across multiple Non-IID levels in federated NLP.
2. **Identification of the FM Fairness Paradox**: FMs worsen worst-client fairness under extreme heterogeneity (α < 0.5) while improving it under moderate heterogeneity (α ≥ 0.5).
3. **FedAvgW investigation**: empirical evidence that aggregation-level fixes are insufficient — the disparity requires dedicated algorithmic solutions.
4. **Critical heterogeneity threshold**: α ≈ 0.5 identified as a practical decision boundary for FM vs. task-specific model selection in cross-silo FL deployments.
5. **Recommendation**: worst-client accuracy should be reported alongside average accuracy in all federated NLP research involving foundation models.

---

## Method

### Models Compared

| Model | Parameters | Type |
|:---|:---:|:---|
| TextCNN (Kim, 2014) | 2.7M | Task-specific lightweight baseline |
| DistilBERT + LoRA (r=8) | 66M total / ~630K trainable | Foundation model + PEFT |

### Federated Setup

- **Algorithm:** FedAvg (McMahan et al., 2017)
- **Clients:** K = 10
- **Rounds:** 50 (α=0.1) / 20 (α ≥ 0.5)
- **Non-IID Partitioning:** Dirichlet Dir(α), α ∈ {0.1, 0.5, 1.0, 5.0}
- **Datasets:** AG News (primary), Sentiment140 (cross-dataset validation)

### FedAvgW: Inverse-Weighted LoRA Aggregation

We investigate whether giving minority clients more aggregation weight resolves the disparity:

$$w_k^{\text{LoRA}} = \frac{(1/n_k)^{\beta}}{\sum_j (1/n_j)^{\beta}}$$

**Result:** No β value improves worst-client accuracy. Stronger weighting (β=0.5) performs *worse* than FedAvg, confirming the problem is semantic incompatibility, not aggregation volume.

---

## Results

### Main Result: FM Fairness Paradox (AG News, K=10, FedAvg)

| α | Model | Avg (%) | Worst (%) | Gap (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.1 | TextCNN | 86.6 | 54.5 | 32.2 |
| 0.1 | DistilBERT+LoRA | 80.8 | 30.7 | **50.1** |
| 0.5 | TextCNN | 95.6 | 91.9 | 3.7 |
| 0.5 | DistilBERT+LoRA | 93.6 | 92.3 | **1.3** |
| 1.0 | TextCNN | 94.9 | 89.3 | 5.6 |
| 1.0 | DistilBERT+LoRA | 93.1 | 93.1 | **0.0** |
| 5.0 | TextCNN | 97.8 | 94.7 | 3.1 |
| 5.0 | DistilBERT+LoRA | 91.3 | 91.3 | **0.0** |

### Cross-Dataset Validation (Sentiment140)

| Model | α | Avg (%) | Worst (%) | Gap (%) |
|:---|:---:|:---:|:---:|:---:|
| TextCNN | 0.1 | 63.2 | 0.0 | 63.2 |
| DistilBERT+LoRA | 0.1 | 61.7 | 18.6 | 43.1 |
| TextCNN | 1.0 | 89.6 | 76.4 | 13.2 |
| DistilBERT+LoRA | 1.0 | 81.1 | 81.1 | 0.0 |

The critical threshold at α ≈ 0.5 replicates across both datasets.

### FedAvgW Results (α=0.1, Round 20)

| Method | Avg (%) | Worst (%) | Gap (%) |
|:---|:---:|:---:|:---:|
| FedAvg | 78.3 | **20.5** | 57.8 |
| FedAvgW β=0.1 | 78.1 | 19.6 | 58.5 |
| FedAvgW β=0.5 | 77.2 | 17.4 | 59.8 |

---

## Why FMs Fail Minority Clients: Global-Local Feature Interference

We hypothesize that this interference mechanism operates as follows: DistilBERT's frozen backbone encodes a general semantic space shaped by pretraining. Under extreme heterogeneity, the majority clients' LoRA gradients push the adapters toward majority-class semantic boundaries in a way that is structurally inconsistent with the minority client's one-class data distribution. Each aggregation step overwrites the minority client's partial adaptation with the majority consensus.

This explains both:
- The **larger gap**: FM adapters are more expressive — they fit the majority distribution more completely, making interference stronger
- The **oscillation**: the minority client achieves partial adaptation during local training before being overwritten at aggregation

Geometric evidence (t-SNE of adapter representations) is planned for the journal extension targeting IEEE Access.

---

## Repository Structure

```
FM-Fairness-Paradox/
├── notebooks/
│   └── FM_Fairness_Paradox_Experiments.ipynb   # Full experimental notebook
├── figures/
│   ├── fig1_fm_fairness_paradox.png             # Main result figure
│   ├── fig2_critical_threshold.png              # Heterogeneity threshold
│   ├── fig3_training_curves.png                 # Training dynamics at α=0.1
│   ├── fig4_sentiment140_validation.png         # Cross-dataset validation
│   └── fig5_fedavgw_comparison.png              # FedAvgW analysis
├── results/
│   └── results_summary.csv                      # All numerical results
├── CITATION.cff                                 # Citation file
├── LICENSE
└── README.md
```

---

## Reproducing Results

All experiments were run on Kaggle (GPU T4 x2). The full notebook is available at:  
**[kaggle.com/kiranmuhammad](https://www.kaggle.com/kiranmuhammad)**

```python
# Key dependencies
pip install torch transformers peft datasets

# Datasets
# AG News: available via HuggingFace datasets ('ag_news')
# Sentiment140: available via HuggingFace datasets ('sentiment140')
```

Detailed reproduction instructions are in the notebook.

---

## Practical Guidance for Practitioners

> **Before deploying a foundation model in a federated setting, measure your data heterogeneity.**
> 
> - If your real-world Dirichlet α is **below 0.5** (common in cross-silo settings with highly specialised institutions such as hospitals), a task-specific model may provide **fairer outcomes** for minority clients.
> - If α is **above 0.5**, the FM's language priors become genuinely protective and it is the fairer choice.

---

## Part of the FORGE Framework

This work is **FORGE-EVAL**, one component of the FORGE research framework:

| Component | Paper | Focus |
|:---|:---|:---|
| FORGE-DIAG | ECCV 2026 (under review) | VLM instability diagnosis |
| FORGE-BENCH | MIUA 2026 (under review) | FL stress-testing benchmark |
| FORGE-EVAL | **This work** | Worst-client evaluation |
| FORGE-ADAPT | In development | Adaptive stabilisation |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{naseer2026fmfairness,
  title     = {When More Parameters Hurt: Foundation Model Priors Amplify 
               Worst-Client Disparity Under Extreme Federated Heterogeneity},
  author    = {Naseer, Kiran and Shoaib, Umar},
  booktitle = {FL@FM Workshop, International Joint Conference on 
               Artificial Intelligence (IJCAI)},
  year      = {2026},
  note      = {arXiv:ARXIV_ID_HERE}
}
```

*Update the arXiv ID once the preprint is posted.*

---

## Contact

**Kiran Naseer**  
PhD Candidate, University of Gujrat, Pakistan  
kirannaseer8@gmail.com  
[LinkedIn](https://linkedin.com/in/kiran-naseer) | [Google Scholar](https://scholar.google.com/citations?user=Ek9e3qwAAAAJ) | [ORCID](https://orcid.org/0009-0005-5129-8155)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
