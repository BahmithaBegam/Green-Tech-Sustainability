# 🌱 GreenTech Sustainability Intelligence & Classifier

An end-to-end Machine Learning and interactive dashboard framework designed to benchmark, evaluate, and predict the environmental sustainability of industrial energy technologies.

---

## 1. Project Overview

### 1.1 Problem Statement
Evaluating clean energy solutions requires balancing emissions, generation capacity, renewability, and operational expenditures[cite: 1]. Manual assessment is inconsistent and slow, making automated, data-driven validation essential for energy audits and green investment decisions.

### 1.2 Proposed Solution
This solution integrates a supervised binary classification pipeline and an interactive user interface:
* **Machine Learning Engine:** Utilizes an optimized Logistic Regression classifier (`lrmodel_sustainable.pkl`) to evaluate the viability of energy systems[cite: 1].
* **Interactive UI:** A multi-tab Streamlit dashboard (`app.py`) providing instant parametric predictions, batch processing, and exploratory data visualizations.

---

## 2. Dataset Specifications

### 2.1 Feature Definitions
The model evaluates four key engineering and economic indicators to determine the target classification[cite: 1]:

| Feature Name | Type | Range / Domain | Unit | Description |
| :--- | :---: | :---: | :---: | :--- |
| `carbon_emissions` | `float64` | `51.9` – `395.4` | $\text{gCO}_2/\text{kWh}$ | Carbon footprint intensity per unit of energy generated[cite: 1]. |
| `energy_output` | `float64` | `106.3` – `987.1` | $\text{MWh}$ | Gross energy output produced by the facility[cite: 1]. |
| `renewability_index` | `float64` | `0.01` – `0.99` | Ratio ($0.0 - 1.0$) | Fraction of input energy derived from renewable resources[cite: 1]. |
| `cost_efficiency` | `float64` | `0.56` – `4.96` | Score Index | Normalized economic expenditure index (lower indicates lower cost overhead)[cite: 1]. |
| **`sustainability`** *(Target)* | `int64` | `{0, 1}` | Binary Flag | `1` = Sustainable / Approved, `0` = Non-Sustainable[cite: 1]. |

### 2.2 Statistical Summary
Summary statistics across the 100 benchmark records in `green_tech_data.csv`[cite: 1]:
* **Total Samples:** 100 observations[cite: 1].
* **Class Distribution:** 85 Non-Sustainable (`0`) and 15 Sustainable (`1`)[cite: 1].
* **Mean Carbon Emissions:** $214.56\text{ gCO}_2/\text{kWh}$[cite: 1].
* **Mean Energy Output:** $548.05\text{ MWh}$[cite: 1].
* **Mean Renewability Index:** $0.52$[cite: 1].
* **Mean Cost Efficiency Index:** $2.71$[cite: 1].

---

## 3. Mathematical Formulation & Model Architecture

### 3.1 Logistic Regression Model
The classifier computes the posterior probability of sustainability using the sigmoid activation function[cite: 1]:

$$P(Y = 1 \mid \mathbf{X}) = \frac{1}{1 + e^{-z}}$$

Where the logit link $z$ is formulated as[cite: 1]:

$$z = \beta_0 + \beta_1(\text{carbon}) + \beta_2(\text{energy}) + \beta_3(\text{renewability}) + \beta_4(\text{cost}) \text{ \cite{1}}$$

### 3.2 Learned Parameters
The serialized model (`lrmodel_sustainable.pkl`) contains the following fitted parameters[cite: 1]:

* **Intercept ($\beta_0$):** `+3.9160`[cite: 1]
* **`renewability_index` ($\beta_3$):** `+1.0922` (Strongest positive driver toward sustainability)[cite: 1]
* **`energy_output` ($\beta_2$):** `+0.0011` (Mild positive driver)[cite: 1]
* **`carbon_emissions` ($\beta_1$):** `-0.0233` (Negative driver; penalizes high emissions)[cite: 1]
* **`cost_efficiency` ($\beta_4$):** `-1.2596` (Negative driver; penalizes high operational cost ratios)[cite: 1]

### 3.3 Model Performance
* **Overall Classification Accuracy:** `94.0%`
* **Non-Sustainable (`Class 0`):** Precision: `0.94`, Recall: `0.99`, F1-Score: `0.97`
* **Sustainable (`Class 1`):** Precision: `0.91`, Recall: `0.67`, F1-Score: `0.77`

---

## 4. Application Architecture & Dashboard Modules

The user interface (`app.py`) is organized into five interactive operational tabs:

### 4.1 Tab 1: Single Assessment
* **Interactive Sliders:** Allows real-time parameter tuning across all four metrics.
* **Outcome Banner:** Instantly displays approval status (`SUSTAINABLE` vs. `NON-SUSTAINABLE`).
* **Confidence Bar Chart:** Renders a horizontal probability distribution comparing both classes.

### 4.2 Tab 2: Batch Inference
* **CSV Drag-and-Drop:** Accepts bulk facility records with automatic schema validation.
* **Predictive Augmentation:** Appends predictions, confidence scores, and categorical status tags to the dataframe.
* **Export Capability:** Provides a download button to export results as `greentech_predictions.csv`.

### 4.3 Tab 3: Data Analytics (EDA)
* **Class Balance Chart:** Visualizes the proportion of sustainable vs. non-sustainable systems.
* **Correlation Heatmap:** Computes and displays Pearson correlation values across all attributes.
* **Distribution Boxplots:** Compares feature variance and outliers grouped by target sustainability class.

### 4.4 Tab 4: Model Diagnostics
* **Coefficient Inspection:** Tabular view of learned model weights and intercept value.
* **Impact Visualizer:** Directional bar chart indicating which factors promote or hinder sustainability compliance.

### 4.5 Tab 5: Documentation
* Outlines operational guidelines, decision thresholds, and underlying mathematical equations.

---

## 5. Repository Structure

```text
├── green_tech_data.csv          # Baseline training dataset (100 records)
├── lrmodel_sustainable.pkl      # Pre-trained scikit-learn model artifact
├── app.py                       # Streamlit web application script
├── requirements.txt             # Environment dependencies
└── README.md                    # Project documentation
