# Manual Statistical Analysis on Clinical Dataset

**Dataset:** [Clinical patient dataset](https://shorturl.at/XRWwW)  
**Language:** Python

---

## Overview

This repository contains a manual, hands-on statistical analysis of a clinical dataset of patients, aimed at reinforcing statistical concepts. The code is intentionally written in a way that exposes the underlying mechanics, using direct implementations rather than relying on high-level libraries. **WARNING:** The MLE and coefficient CI code is inefficient and computationally heavy, primarily due to the use of for-loops and the sympy library.

Despite the inefficiencies and some inaccuracies, this project was a valuable learning experience, helping me grow my practical and theoretical statistical skills. I thoroughly enjoyed working on it!

*I plan to expand and refine this analysis in the future.*

---

## Methods Used

- **Data Cleaning:**  
  - Removed all `-9` (unidentified/bad) diagnoses from `mh1`, `mh2`, and `mh3`, storing valid entries in a list.

- **Descriptive Statistics:**  
  - Calculated frequencies for each diagnosis and matched them with the cleaned data.

- **Inferential Statistics:**  
  - **Pearson’s r:**  
    - Tested associations between variables. Found a significant association between Bipolar Disorder and "Other (unspecified)" disorder (null hypothesis rejected at α = 0.05).
  - **Chi-Squared Test:**  
    - Attempted for Bipolar Disorder, but the test conditions were not satisfied for all comparisons.
    - Performed on alternate contingency tables to ensure independence, e.g., BPD diagnosis vs. other variables.
  - **Binary Logistic Regression:**  
    - Built a model to predict the probability of a BPD diagnosis in the second diagnosis period based on previous diagnosis and patient information.
    - Manually implemented MLE for Bernoulli’s πₙ parameter using gradient ascent.
    - Calculated coefficient confidence intervals using the Hessian matrix.
  - **Cochran-Mantel-Haenszel (CMH) Test:**  
    - Compared SMI/SED vs. Substance Abuse strata by diagnosis.
    - Examined Substance Abuse vs. Employment strata by SMI/SED.

---

## Log

- Cleaned data by removing all `-9` (unidentified/bad) diagnoses from `mh1`, `mh2`, `mh3`.
- Found frequency counts for each diagnosis.
- Performed Pearson’s r test for associations; found significant association for Bipolar Disorder and "Other (unspecified)" disorder (α = 0.05).
- Attempted chi-squared test for Bipolar Disorder, but conditions were not satisfied.
- Built a binary logistic regression model to estimate probability of BPD diagnosis in the second period, given patient info and past diagnosis.
- Used alternate contingency tables for chi-squared tests to ensure variable independence.
- Ran CMH tests for:
    - SMI/SED vs. Substance Abuse (stratified by diagnosis)
    - Substance Abuse vs. Employment (stratified by SMI/SED)

---

## Reflections

This project is intentionally “manual” and not optimized, but it was very fun and a great way to reinforce statistical concepts. There are clear inefficiencies and likely some inaccuracies, but every challenge was a learning opportunity. I look forward to refining and expanding this project in the future! :)

---
