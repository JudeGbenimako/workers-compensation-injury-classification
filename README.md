# Workers' Compensation Claim Injury Type Classification

## Overview

This academic machine learning project explored how historical workers' compensation claims could support the classification of **Claim Injury Type** for the New York Workers' Compensation Board (WCB). The goal was to develop a multiclass classification model and a user-friendly prototype for exploring its predictions.

Developed as a group project for the MSc in Data Science and Advanced Analytics at **NOVA Information Management School (NOVA IMS)**, 2024/2025.

**This is a proof-of-concept for decision support, not a deployed WCB system or an automated determination of benefits eligibility.**

## Dataset and task

The original training dataset contained **593,471 observations and 33 columns**, representing workers' compensation claims from 2020–2022. The target variable, **Claim Injury Type**, contained **eight classes**. The target was imbalanced.

The raw training and test datasets are **not included** in this portfolio repository.

## Project workflow

1. **Data exploration and cleaning:** Investigated duplicates, missing fields, inconsistent values, and outliers.
2. **Preprocessing and feature engineering:** Used group-based imputation, categorical encoding, scaling, and engineered variables, including time-to-assembly and form-submission indicators.
3. **Feature selection:** Combined correlation analysis, Random Forest feature importance, mutual information, and Lasso-based assessment. The selected feature set contained **16 predictors**.
4. **Model comparison:** Evaluated seven classifiers, including decision trees, Random Forest, histogram gradient boosting, a multilayer perceptron, AdaBoost, voting, and stacking ensembles.
5. **Model optimization:** Tuned the selected StackingClassifier, combining Random Forest and HistGradientBoosting base estimators with an MLP final estimator.
6. **Prototype interface:** Used Streamlit to accept claim features and display a predicted injury class, model-assigned probability, and feature importance from the Random Forest base estimator.

## Reported results

| Metric | Reported result |
|---|---:|
| Validation strategy | Stratified 80/20 holdout |
| Primary evaluation metric | Weighted F1-score |
| Selected model | Optimized StackingClassifier |
| **Validation weighted F1** | **0.7255** |
| Selected predictors | 16 |
| Target classes | 8 |

Weighted F1-score was used for model comparison because the classes were imbalanced. The reported **0.7255** is a validation result, not a production performance guarantee.

## Repository files

- [01_data_exploration_preprocessing.ipynb](01_data_exploration_preprocessing.ipynb) — Exploratory analysis, cleaning, encoding, and scaling.
- [02_model_development_evaluation.ipynb](02_model_development_evaluation.ipynb) — Feature selection, classifier comparison, tuning, and prediction workflow.
- [03_streamlit_interface.ipynb](03_streamlit_interface.ipynb) — Notebook that generates the Streamlit app.
- [app.py](app.py) — Streamlit application source.
- [requirements.txt](requirements.txt) — Python dependencies.
- [.gitignore](.gitignore) — Exclusion rules for local data and model artifacts.

The notebooks have been **cleared of saved execution outputs** and student ID/contact information for portfolio presentation. Code and modeling choices remain those of the original academic project.

## Running the prototype

Install the dependencies:

```bash
pip install -r requirements.txt
```

The analysis notebooks require the original `train_data.csv` and `test_data.csv` and create intermediate datasets and trained model artifacts. These files are **not provided here**.

The Streamlit app references four trained artifacts:

- `final_model.pkl`
- `scaler.pkl`
- `target_encoder.pkl`
- `label_encoder.pkl`

Once compatible, trusted artifacts have been generated and placed alongside `app.py`, the interface can be started with:

```bash
streamlit run app.py
```

**The application cannot run from this repository alone:** the original datasets and serialized model artifacts are deliberately omitted. Do not load untrusted `.pkl` or other pickle-based model files.

## Limitations and responsible use

- Class imbalance can conceal poor performance on minority injury types even when weighted F1 is satisfactory.
- The probability shown in the app is the highest `predict_proba` value and is **not established as a calibrated confidence measure**.
- The feature-importance plot is from the Random Forest component, not a complete explanation of the stacked ensemble.
- The random holdout assesses the reported classification experiment but does not establish performance on future claims under changing conditions.
- Any real decision-support use would require independent validation, fairness and privacy assessments, monitoring, and human review.

## Team

**Group 29, NOVA IMS:** Inês Honrado, Jude Gbenimako, Ruben Marques, Susana Reis, and Tomás Carvalho.

The work was collaborative. Jude contributed literature research and participated in comparing and evaluating classification models. The Streamlit interface was developed by Inês and Susana.

**Original team project:** [Machine Learning Group Project — Group 29](https://github.com/inesph/Machine-Learning-Group-Project---Group-29)

## Academic context

- **Institution:** NOVA Information Management School (NOVA IMS)
- **Program:** MSc in Data Science and Advanced Analytics
- **Course:** Machine Learning
- **Academic year:** 2024/2025
