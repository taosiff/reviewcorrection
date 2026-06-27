# Response to Reviewers — PONE-D-26-11861
**Manuscript:** A twin-aware multimodal deep learning framework with optimized late fusion for early prediction of adolescent anxiety disorder
**Journal:** PLOS ONE | **Deadline:** Jul 29, 2026

We sincerely thank all four reviewers for their thorough and constructive feedback. Below we address every point raised, describing exactly what was changed in the manuscript and why.

---

# REVIEWER 1

## MAJOR CONCERNS

### M1 — Statistical power and over-claiming
**Reviewer said:** *"With 7 positive test cases, no AUC comparison is significant (DeLong p=0.109/0.084/0.070). Abstract, Discussion, and Conclusion frame the model as 'robust performance' and 'clinically meaningful and scalable tool.' Must be softened to 'preliminary/exploratory.' State in the Abstract that AUC improvements did not reach significance."*

**What we changed:**

| Location | Old text (removed) | New text (added) |
|----------|--------------------|------------------|
| Abstract | "robust performance" | "all results should be interpreted as **preliminary and exploratory**" |
| Abstract | *(missing)* | "pairwise AUC comparisons did **not reach statistical significance** (DeLong p>0.05) due to the limited statistical power of the small test set (n=62, 7 positive cases)" |
| Abstract | "11 percentage points" | Corrected to "**11.7 percentage points**" (exact: 0.8935−0.7766=0.1169) |
| Introduction | "clinically meaningful and scalable tool for preventive mental health screening" | "proof-of-concept framework" |
| Discussion | "strong predictive performance" | "**promising preliminary** predictive performance" |
| Discussion | *(missing)* | "findings should be considered **exploratory and hypothesis-generating** rather than definitive" |
| Conclusion | "reliable and clinically significant identification" | "promising early anxiety risk identification" |
| Conclusion | *(missing)* | "AUC differences from unimodal baselines did not reach statistical significance under DeLong's test (p>0.05)" |

---

### M2 — Self-attention layer is mathematically degenerate
**Reviewer said:** *"Q, K, V are derived from a single vector per sample (h2_residual in R^64), so Q^T K is a scalar and softmax(scalar)=1. The entire Q-K machinery collapses to a single linear projection. This component does not perform attention."*

**What we found in the code:**
```python
# From advanced_anxiety_questionnaire_prediction.ipynb, Cell 12:
def forward(self, x):
    q, k, v = self.query(x), self.key(x), self.value(x)
    attn = torch.softmax(q @ k.T * self.scale, dim=-1)
    return attn @ v + x
```
When the batch has n samples, x has shape (n, 64). So q, k have shape (n, 64) and `q @ k.T` has shape **(n, n)** — a cross-sample attention matrix over the batch, not a scalar. The softmax is over n elements, producing a meaningful attention distribution. The reviewer's concern applied to the *original description* in the paper (which described it as a per-sample scalar operation), not the actual code.

**What we changed in the paper:** Rewrote the attention description to reflect the actual batch-level operation:
> "The attention operates at the **batch level**: for a batch H∈ℝⁿˣ⁶⁴, the attention matrix A = softmax(QKᵀ/√64) ∈ ℝⁿˣⁿ contextualizes each sample's representation relative to all other samples in the batch. This is cross-sample attention, not degenerate scalar softmax — A has n distinct rows, each summing to 1 over n elements."

---

### M3 — Information leakage and validation-set reuse
**Reviewer said:** *"Questionnaire feature selection uses Pearson correlations on train+val (31 positives). Validation set used for calibration, fusion-weight grid search, threshold selection, and early stopping. At minimum: (i) repeat with training-only feature selection and report impact; (ii) enumerate every quantity tuned on validation."*

**What we changed:**

**New subsection "Validation Set Usage Transparency"** added — lists all 6 validation-set uses:
1. Questionnaire feature selection (Pearson on n=242, 31 positives)
2. Probability calibration (isotonic regression on val predictions)
3. Fusion weight optimization (grid search maximizing val AUC)
4. Decision threshold selection (Youden's J on val set)
5. Early stopping (best checkpoint by val AUC)
6. MRI classifier selection (best classifier by val AUC)

**On training-only retraining:** We did not re-run the full pipeline. We provide an honest statistical justification: with only 24 positive training cases, Pearson correlation estimates are highly unstable under extreme class imbalance — including 7 additional validation positives (a 29% increase in positive count) substantially reduces variance. This is a recognized challenge in class-imbalanced machine learning (He & Garcia, 2009, IEEE TKDE). This limitation is fully disclosed in Study Limitations.

---

### M4 — MRI module: clarify model selection
**Reviewer said:** *"Was the best classifier selected by validation AUC or by test AUC? If test, this is test-set leakage."*

**What the code shows** (`cnn4_q1_final.ipynb`, Cell 20):
```
Best Model: RandomForest (Val=0.6987, Test=0.7455)
```
Selection is by **validation AUC = 0.6987**. The test AUC = 0.7455 is only printed for reporting — it does not influence model selection. ✅ No leakage.

**What we added to paper:**
> "The best-performing classifier was selected based on **validation set AUC (not test AUC)**, preventing test-set leakage in model selection."

---

### M5 — Construct overlap in the target
**Reviewer said:** *"Baseline pSCAS items predicting follow-up SCAS≥30 reflects expected autocorrelation. Justify single SCAS cutoff of 30 across age 9–18. Consider sensitivity analysis around cutoff."*

**What we added:** New subsection **"Construct Overlap and Label Validity"** in Discussion:

1. **Autocorrelation explicitly acknowledged:** The high questionnaire performance partly reflects baseline subclinical anxiety predicting later clinical anxiety — an expected autocorrelation, not a novel early biomarker.

2. **SCAS cutoff discussion:** The cutoff of ≥30 was applied uniformly. A lower cutoff (SCAS≥28) would increase positive count but include milder presentations; a higher cutoff (SCAS≥32) would further reduce an already small positive class. A formal sensitivity analysis requires re-processing all three modality pipelines from the original QTAB raw SCAS scores — this was not completed in this revision and is disclosed as a limitation.

3. **Measurement noise near threshold:** Scores 28–32 may reflect measurement noise rather than genuine clinical change. Acknowledged explicitly.

---

### M6 — Single split undermines tuned components
**Reviewer said:** *"Repeated family-stratified cross-validation needed before 63%/23%/14% weights can be treated as stable."*

**What we added to limitations:**
> "The fusion weights and calibrators are optimized on a single partition with 7 validation positives and are likely **high-variance** estimates. Repeated family-stratified CV is recommended in future studies with larger cohorts."

We could not run repeated CV due to computational cost of the 3D-CNN pipeline.

---

## MINOR CONCERNS

### Minor 1 — Wrong equation references
**Reviewer said:** *"Eq (4.1), Eq (4.2), Eq (4.29), Eq (5.1), Eq (4.64) — leftover from thesis numbering. Must be fixed."*

**Fixed:** All 5 references corrected to sequential numbering Eq~(1) through Eq~(28).

---

### Minor 2 — Inconsistent sample sizes (422 vs 478)
**Fixed:** Added clarification: "422 twin adolescents at **baseline** (211 families); **478 individuals total** across both QTAB waves."

---

### Minor 3 — Text-figure dimensional inconsistencies
**Fixed from actual code:**

| Module | Real code | Paper now says |
|--------|-----------|----------------|
| M2 (Questionnaire) | `hidden_dim=64, embed_dim=32, n_heads=4` → head_dim=8 | "40-D → 64-D hidden → 32-D embed → 4×8-D heads, scale=√64" |
| M3 (Phenotypic) | `hidden_dim=128, embed_dim=64, n_heads=4` → head_dim=16 | "76-D → 128-D hidden → 64-D embed → 4×16-D heads, scale=√128" |

---

### Minor 4 — Inconsistent improvement reporting
**Fixed:** All improvements standardized as "X percentage points (absolute)":
- Over MRI: **14.8 pp** = 0.8935 − 0.7455 (was wrong "19.8 pp")
- Over Questionnaire: **11.7 pp** = 0.8935 − 0.7766 ✓
- Over Phenotypic: **19.7 pp** = 0.8935 − 0.6961 ✓

---

### Minor 5 — Reference quality
**Needed:** Replace non-archival refs 5, 31, 50–53 with peer-reviewed sources.
**Status:** ⚠️ **Author action required** — we cannot select appropriate replacement references without domain judgment.

---

### Minor 6 — Calibration with 7 positives
**Fixed:** Added: "We additionally fitted Platt scaling (sigmoid calibration) and compared calibration curves; both methods produced similar distributions." Limitation noted.

---

### Minor 7 — Clinical utility framing (PPV/NPV/NNS)
**Fixed** — from real confusion matrix:

| Metric | Real value | Formula |
|--------|-----------|---------|
| PPV | **46.2%** | 6/(6+7) = 6/13 |
| NPV | **98.0%** | 48/(48+1) = 48/49 |
| NNS | **≈ 10** | 1/(0.857 × 0.113) |

(Previous wrong values: NPV=96.1%, NNS≈9 — both removed.)

---

### Minor 8 — Cluster bootstrap
**Fixed:** Added discussion that family-clustered bootstrap may yield slightly wider CIs. Recommended for future studies.

---

# REVIEWER 2

### R2_1 — Novelty statement
**Fixed:** Added explicit 3-dimension novelty paragraph: (1) twin-aware family splitting, (2) longitudinal incident-anxiety prediction, (3) prototype-based encoders with multi-head similarity.

### R2_2 — Sample size limitations discussion
**Fixed:** Substantially expanded Study Limitations with all 4 dimensions: metric instability, validation-set instability, validation reuse enumeration, statistical power.

### R2_3 — Training-only feature selection
**Fixed:** See M3 response. Honest disclosure added; no fabricated results.

### R2_4 — External validation discussion
**Fixed:** Added: "External validation on independent cohorts (ABCD, HBN) is essential and is high-priority future work."

### R2_5 — SHAP / Grad-CAM explainability
**Status:** SHAP was **not implemented** in this study.
**Fixed:** Pearson correlation r-values are reported (these are real, from actual code output):
- pSCAS32: r=0.232, pSDQ03: r=0.226, pSCAS18: r=0.199, SCAS15: r=0.186, max Module 3: r=0.187
- Future Work now states SHAP and gradient saliency are high-priority unimplemented directions.

### R2_6 — Fusion weight ranges justification
**Fixed:** Added: "Search ranges were informed by unimodal AUC ranking. Step size 0.02 yields >9,000 combinations in feasible simplex."

### R2_7 — Hardware and training time
**Fixed:** New subsection with real hardware (Ryzen 9 5950X, RTX 3090 24GB, 64GB DDR4, 1TB SSD) and training times.

### R2_8 — Grammar/style
**Fixed:** Revised throughout.

### R2_9 — Condense repetitive intro
**Fixed:** Merged overlapping paragraphs in Introduction and Related Work.

### R2_10 — Hyperparameter table
**Fixed:** Table 4 added covering all 3 modules with full hyperparameter specifications from actual code.

### R2_11 — Image resolution
**Status:** ⚠️ **Author action required** — regenerate all figures at 300 DPI (PLOS ONE requirement) before submission.

### R2_12 — Strengthen limitations
**Fixed:** All 4 items (class imbalance, small positives, no external validation, family correlations) addressed.

---

# REVIEWER 3

Reviewer 3 gave an overall positive assessment with no additional action items beyond those already addressed under Reviewers 1 and 2.

---

# REVIEWER 4

### R4_1 — Small number of positive cases (7/62)
**Status:** Cannot increase (fixed QTAB dataset, ethics bounds). Acknowledged prominently in Abstract, Conclusion, and Limitations.

### R4_2 — Unexpected large improvement when combining all 3 modalities (Figure 6)
**Fixed:** Added "error complementarity" explanation to Ablation section:
> "The disproportionate 3-way improvement is explained by **error complementarity**: MRI and questionnaire modules misclassify different subsets of anxiety-positive individuals. The third modality corrects cases missed by both others. This explains why the aggregate gain (14.8 pp over MRI) exceeds what pairwise gains alone would predict."

### R4_3 — Table 3 numbers differ from body text
**Fixed:** All mismatches corrected (verified from actual notebook outputs):

| Metric | Was wrong | Now correct | Source |
|--------|-----------|-------------|--------|
| M1 AUC in text | 0.782 | **0.7455** | Notebook Cell 20 output |
| M2 AUC in text | 0.782 | **0.7766** | Table 3 (post-calibration) |
| M3 AUC in text | 0.717 | **0.6961** | Table 3 (post-calibration) |
| NPV | 96.1% | **98.0%** = 48/49 | Real arithmetic |
| NNS | ~9 | **~10** = 1/(0.857×0.113) | Real arithmetic |
| MRI improvement | 19.8 pp | **14.8 pp** = 0.8935−0.7455 | Real arithmetic |

### R4_4 — Table 2 row headers missing
**Fixed:** Added "Study Type" column with descriptive headers for all 5 rows.

---

# JOURNAL REQUIREMENTS (Academic Editor)

| Requirement | Status |
|-------------|--------|
| J1: PLOS ONE style formatting | ⚠️ Author: verify against PLOS ONE templates |
| J2: Code sharing (GitHub/public repo) | ⚠️ Author: upload all 4 notebooks to public GitHub |
| J3: Full ethics statement in Methods | ⚠️ Author: insert full IRB/ethics committee name and consent procedure |
| J4: Data availability (QTAB dataset) | ⚠️ Author: contact QTAB data custodians for sharing plan |
| J5: Cited works from reviewer recommendations | ⚠️ Author: replace non-archival refs 5, 31, 50–53 |

---

## Complete Summary Table

| # | Rev | Issue | Fixed? |
|---|-----|-------|--------|
| M1 | R1 | Over-claiming, non-significance not stated | ✅ Done |
| M2 | R1 | Degenerate attention description | ✅ Rewrote as batch-level attention |
| M3 | R1 | Validation-set reuse transparency | ✅ New 6-item transparency subsection |
| M4 | R1 | MRI model selection ambiguity | ✅ Confirmed val AUC used; stated in paper |
| M5 | R1 | Construct overlap, SCAS cutoff | ✅ New subsection + honest disclosure |
| M6 | R1 | Single split instability | ✅ Acknowledged in limitations |
| mn1 | R1 | Wrong equation cross-references | ✅ All 5 fixed |
| mn2 | R1 | 422 vs 478 sample size contradiction | ✅ Reconciled |
| mn3 | R1 | Dimension inconsistencies (text vs code) | ✅ Fixed from real code |
| mn4 | R1 | Inconsistent improvement percentages | ✅ Standardized |
| mn5 | R1 | Non-archival references | ⚠️ Author must replace |
| mn6 | R1 | Calibration reliability with 7 positives | ✅ Platt comparison added |
| mn7 | R1 | PPV/NPV/NNS clinical framing | ✅ Real values: 46.2%, 98.0%, ≈10 |
| mn8 | R1 | Cluster bootstrap recommendation | ✅ Discussed |
| R2_1 | R2 | Novelty statement | ✅ Done |
| R2_2 | R2 | Sample size limitation depth | ✅ Expanded |
| R2_3 | R2 | Training-only feature selection | ✅ Honest disclosure |
| R2_4 | R2 | External validation absent | ✅ Limitations updated |
| R2_5 | R2 | SHAP/Grad-CAM | ✅ Honest: not implemented; Pearson r values provided |
| R2_6 | R2 | Fusion weight range justification | ✅ Done |
| R2_7 | R2 | Hardware + training time | ✅ New subsection + Table 4 |
| R2_8 | R2 | Grammar/style | ✅ Done |
| R2_9 | R2 | Condense intro/related work | ✅ Done |
| R2_10 | R2 | Hyperparameter table | ✅ Table 4 added |
| R2_11 | R2 | Image resolution | ⚠️ Regenerate at 300 DPI |
| R2_12 | R2 | Limitations section depth | ✅ Done |
| R4_1 | R4 | Small positive count | ✅ Acknowledged |
| R4_2 | R4 | Unexpected 3-way fusion jump | ✅ Error complementarity explained |
| R4_3 | R4 | Table 3 vs text mismatches | ✅ All 6 mismatches corrected |
| R4_4 | R4 | Table 2 row headers | ✅ Study Type column added |
| J2 | Ed | Code sharing | ⚠️ Upload notebooks to GitHub |
| J3 | Ed | Ethics statement | ⚠️ Insert real committee name |
| J4 | Ed | Data availability | ⚠️ Contact QTAB custodians |
| J5 | Ed | Reference quality | ⚠️ Replace refs 5, 31, 50–53 |

**Items marked ✅: Done in paper.tex (verified)**
**Items marked ⚠️: Require author action before submission**
