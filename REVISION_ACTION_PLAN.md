# PLOS ONE Revision Action Plan

## Executive Summary

Your manuscript received 4 reviewer responses with a "revise and resubmit" decision. The reviews identify both critical issues that MUST be fixed and minor improvements. This document provides a complete action plan.

## Critical Issues (MUST FIX)

### 1. **Degenerate Self-Attention Layer** (Reviewer #1, Major Concern M2)
**Problem:** The attention mechanism in Equations (8)-(12) is mathematically incorrect. With Q, K, V derived from a single vector per sample, Q^T*K is scalar and softmax(scalar) = 1, making the entire attention layer collapse to just a linear projection.

**Action Required:**
- [ ] EITHER: Fix the attention to do something non-trivial (feature-wise gating, multi-head, or cross-modal attention)
- [ ] OR: Remove the "attention" framing and accurately describe it as "linear projection with residual gating"
- [ ] Update equations and text descriptions
- [ ] Update code in notebooks

### 2. **Feature Selection Leakage** (Reviewer #1, Major Concern M3)
**Problem:** Questionnaire feature selection uses Pearson correlations computed on train+val data, leaking validation labels.

**Action Required:**
- [ ] Re-implement feature selection using ONLY training data
- [ ] Alternatively, use nested cross-validation
- [ ] Re-run experiments and update all results
- [ ] Document the impact on performance

### 3. **MRI Model Selection Ambiguity** (Reviewer #1, Major Concern M4)
**Problem:** Page 9 says Random Forest was "best performing" with test AUC=0.745, but also says evaluation was "on validation set." This is confusing and could indicate test-set leakage.

**Action Required:**
- [ ] Clarify whether selection used validation AUC or test AUC
- [ ] If test AUC was used, this is leakage - must reselect using only validation
- [ ] Update text to be crystal clear about the selection process

### 4. **Statistical Over-claiming** (Reviewer #1, Major Concern M1)
**Problem:** With only 7 positive test cases, AUC improvements are not statistically significant (DeLong p>0.05), but Abstract/Discussion claim "robust performance" and "clinically meaningful and scalable tool."

**Action Required:**
- [ ] Change Abstract to say AUC improvements "did not reach statistical significance"
- [ ] Replace "robust" with "preliminary" or "exploratory" throughout
- [ ] Remove or soften claims about "scalable tool" and "clinical readiness"
- [ ] Emphasize this is a proof-of-concept needing validation

### 5. **Equation Reference Errors** (Reviewer #1, Minor #1)
**Problem:** Text cites "Eq (4.1)", "Eq (4.2)", "Eq (4.29)", "Eq (5.1)", "Eq (4.64)" but equations are numbered (1)-(39).

**Action Required:**
- [ ] Search and replace all equation references in paper.tex
- [ ] Ensure every reference matches actual equation numbers

### 6. **Dimensional Inconsistencies** (Reviewer #1, Minor #3)
**Problem:** Text descriptions don't match Figure 4 for Module 2 and Module 3 architectures.

**Action Required:**
- [ ] For Module 2: Reconcile 40-D vs 76-D input, 64-D vs 32-D embedding, 8-D vs 16-D heads, sqrt(64) vs sqrt(128)
- [ ] For Module 3: Match text and figure
- [ ] Either update text OR regenerate Figure 4

### 7. **Table 3 Number Mismatches** (Reviewer #4, Issue #3)
**Problem:** M1 ROC-AUC is 0.7455 in Table 3 but 0.782 in text (line 415). Multiple mismatches suspected.

**Action Required:**
- [ ] Audit ALL numbers in Table 3 against text
- [ ] Fix discrepancies
- [ ] Add explanation if differences are intentional (e.g., train vs validation vs test)

## Important Improvements (SHOULD FIX)

### 8. **Sample Size Inconsistency** (Reviewer #1, Minor #2)
**Problem:** Methods say "422 twin adolescents" but Limitations say "n=304 out of 478 total."

**Action Required:**
- [ ] Clarify the participant flow: 478 baseline → 422 with data → 304 with complete multimodal data
- [ ] Add a participant flow diagram if helpful

### 9. **Inconsistent Performance Reporting** (Reviewer #1, Minor #4)
**Problem:** Improvements reported as "11 pp", "11.74 pp", "+15.1%", "11.4%" inconsistently.

**Action Required:**
- [ ] Choose ONE convention (recommend: absolute percentage points)
- [ ] Update all mentions to use same number

### 10. **Replace Non-Peer-Reviewed References** (Reviewer #1, Minor #5)
**Problem:** Refs 50-53 are ScienceDirect "topics" pages, not primary sources. Refs 5, 31 also problematic.

**Action Required:**
- [ ] Replace ref 5 (ResearchGate)
- [ ] Replace ref 31 (labsolver page)
- [ ] Replace refs 50-53 (grid search, Youden index, multimodal fusion) with textbooks or primary papers

### 11. **Add Table 2 Row Headers** (Reviewer #4, Issue #4)
**Problem:** Table 2 has no row headers/types.

**Action Required:**
- [ ] Add a column or section headers to categorize the rows in Table 2

### 12. **Explain Figure 6 Better** (Reviewer #4, Issue #2)
**Problem:** Figure 6 shows pairwise combinations give small gains but 3-way combination gives huge gain. This is surprising and poorly explained.

**Action Required:**
- [ ] Add 2-3 paragraphs discussing why 3-way fusion outperforms pairwise
- [ ] Possible explanations: complementary information, optimized weights, calibration effects
- [ ] Reference this in Discussion section

### 13. **Discuss Construct Overlap** (Reviewer #1, Major Concern M5)
**Problem:** Target (child SCAS ≥30 at follow-up) is predicted from baseline SCAS items, which is expected autocorrelation, limiting novelty.

**Action Required:**
- [ ] Add explicit discussion of this in Results/Discussion
- [ ] Acknowledge that strong questionnaire performance reflects baseline→follow-up symptom continuity
- [ ] Justify SCAS cutoff of 30 across 9-18 age range and both sexes
- [ ] Consider discussing label noise near threshold (29→31 might be measurement noise)

### 14. **Add Clinical Utility Metrics** (Reviewer #1, Minor #7)
**Problem:** At 11.3% prevalence and precision 0.46, ~50% of flagged adolescents are false positives. Paper emphasizes AUC/F1 but not PPV/NPV.

**Action Required:**
- [ ] Calculate and report Positive Predictive Value (PPV) and Negative Predictive Value (NPV)
- [ ] Discuss number-needed-to-screen
- [ ] Address false positive rate in clinical context

## Recommended Improvements (NICE TO HAVE)

### 15. **Repeated Cross-Validation** (Reviewer #1, Major Concern M6)
**Problem:** Study uses single data split. Fusion weights and performance might be unstable.

**Action Required (if feasible):**
- [ ] Implement repeated family-stratified k-fold cross-validation
- [ ] Report mean and standard deviation of performance metrics
- [ ] Show stability of fusion weights across folds
- **Note:** Reviewers acknowledge this is computationally expensive - if not feasible, strengthen discussion of single-split limitation

### 16. **Add Explainability Analyses** (Reviewer #2, Issue #5)
**Action Required:**
- [ ] Add SHAP values for feature importance
- [ ] Add Grad-CAM visualizations for MRI module
- [ ] Add prototype interpretation examples for Quest/Pheno modules

### 17. **Add Hyperparameter Table** (Reviewer #2, Issue #10)
**Action Required:**
- [ ] Create comprehensive table listing all hyperparameters
- [ ] Include learning rates, batch sizes, epochs, architecture details, optimizer settings

### 18. **Add Computational Details** (Reviewer #2, Issue #7)
**Action Required:**
- [ ] Document hardware (GPU model, RAM, etc.)
- [ ] Report training times for each module
- [ ] Estimate computational complexity

### 19. **Improve Image Quality** (Reviewer #2, Issue #11)
**Action Required:**
- [ ] Regenerate all figures at higher resolution
- [ ] Ensure figures are crisp and readable when printed

### 20. **Writing Quality** (Reviewers #1, #2)
**Action Required:**
- [ ] Proofread for grammar, typos, style
- [ ] Condense repetitive sections in Introduction and Related Work
- [ ] Strengthen Limitations section (mention class imbalance, small positives, no external validation, family correlations)

## PLOS ONE Format Requirements

### 21. **Code Sharing**
**Action Required:**
- [ ] Prepare code for public release (GitHub/GitLab)
- [ ] Clean up notebooks with clear documentation
- [ ] Include README with setup instructions
- [ ] Add requirements.txt or environment.yml
- [ ] **Important:** Code must be available WITHOUT restrictions upon publication

### 22. **Ethics Statement**
**Action Required:**
- [ ] Add full ethics statement to Methods section
- [ ] Include full name of IRB/ethics committee
- [ ] State whether informed written/verbal consent was obtained
- [ ] If consent waived, include that information

### 23. **Data Availability**
**Action Required:**
- [ ] Clarify data sharing plan
- [ ] QTAB data is restricted - explain access process
- [ ] Note that you cannot make QTAB data freely available due to ethics/privacy

### 24. **Manuscript Formatting**
**Action Required:**
- [ ] Follow PLOS ONE LaTeX style templates
- [ ] Check file naming conventions
- [ ] Verify formatting matches sample documents

## Submission Requirements

### 25. **Response to Reviewers Document**
**Action Required:**
- [ ] Create separate "Response to Reviewers" letter
- [ ] Address EVERY point from all 4 reviewers
- [ ] Use format: "Reviewer X, Comment Y: [quote] → Response: [your response]"
- [ ] Reference specific line numbers or sections where changes were made

### 26. **Track Changes Version**
**Action Required:**
- [ ] Create "Revised Manuscript with Track Changes"
- [ ] Use LaTeX changes package or similar
- [ ] Highlight all modifications

### 27. **Clean Final Version**
**Action Required:**
- [ ] Create unmarked "Manuscript" file
- [ ] No track changes, clean final version

## Priority Order

### Phase 1: Critical Fixes (Must complete before resubmission)
1. Fix degenerate attention layer (Issue #1)
2. Fix feature selection leakage (Issue #2)
3. Clarify MRI model selection (Issue #3)
4. Soften statistical claims (Issue #4)
5. Fix equation references (Issue #5)
6. Fix dimensional inconsistencies (Issue #6)
7. Fix Table 3 mismatches (Issue #7)

### Phase 2: Important Improvements
8-14 (All the "Should Fix" items)

### Phase 3: Recommended Improvements
15-20 (As time and resources allow)

### Phase 4: Formatting and Submission
21-27 (Final preparation)

## Code Files to Update

Based on the notebook names in your folder:
- `advanced_anxiety_questionnaire_prediction.ipynb` - Module 2 (Questionnaire), fix attention, feature selection
- `advanced_static_prediction.ipynb` - Module 3 (Phenotypic), fix attention if present
- `cnn4_q1_final.ipynb` or `cnn4_q1_final_with_extraction.ipynb` - Module 1 (MRI), clarify model selection
- All notebooks: Update to address leakage issues, add explainability

## LaTeX File to Update

- `paper.tex` - Main manuscript file

## Timeline Suggestion

You have until **July 29, 2026** to submit revisions.

- **Weeks 1-2:** Complete Phase 1 (Critical fixes in code and paper)
- **Weeks 3-4:** Complete Phase 2 (Important improvements)
- **Week 5:** Complete Phase 3 (Nice-to-haves as time allows)
- **Week 6:** Complete Phase 4 (Formatting and submission documents)
- **Buffer:** 2 weeks for unexpected issues

## Questions to Ask Your Team

1. Can you access more QTAB data to increase sample size? (Reviewer #4 concern)
2. Is repeated cross-validation computationally feasible? (Reviewer #1 recommendation)
3. Do you have computational resources for full re-training after fixing leakage?
4. Which team member will handle which notebook?
5. Who will write the Response to Reviewers letter?

## Summary

**Critical Issues:** 7 MUST-FIX items (attention layer, leakage, claims, references, tables)
**Important Issues:** 7 SHOULD-FIX items (explanations, metrics, documentation)
**Nice-to-Have:** 6 improvements (explainability, cross-validation, etc.)
**Format Requirements:** 7 items (code sharing, ethics, submission docs)

**Total:** 27 action items across 4 priority phases

The reviews are overall positive - Reviewer #3 says it's "scientifically sound" and Reviewer #4 welcomes the methodology. The main concerns are:
1. Small sample size (acknowledged but can't fix)
2. Technical correctness issues (CAN and MUST fix)
3. Statistical over-claiming (MUST soften language)
4. Presentation clarity (CAN improve)

Good luck with your revisions! The framework is solid - you mainly need to fix technical issues and moderate your claims.
