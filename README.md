# WiDS Datathon 2025 — Break Through Tech AI Project

## 👥 Team Members

- Madison Harman – [@MadisonHarman](https://github.com/MadisonHarman)
- Ananya Aatreya - [@ananyaa06](https://github.com/ananyaa06)
- Amanda Yu – [@amandayu255](https://github.com/amandayu255)

---

## 🧠 Project Overview

### About the Competition  
The **WiDS Datathon 2025** is a machine learning competition hosted on Kaggle in collaboration with the **Ann S. Bowers Women’s Brain Health Initiative (WBHI)**, **Cornell University**, and **UC Santa Barbara**. The datasets and support are provided by the **Healthy Brain Network (HBN)**, the signature scientific initiative of the **Child Mind Institute**, and the **Reproducible Brain Charts (RBC)** project.

This datathon is part of the **Break Through Tech AI Program**, designed to give women and nonbinary students hands-on experience in machine learning while contributing to a meaningful real-world problem in neuroscience and mental health.

**Kaggle competition link:** [WiDS Datathon 2025](https://www.kaggle.com/competitions/widsdatathon2025)

### Challenge Objective  
Participants are tasked with building a machine learning model to **predict an individual’s ADHD diagnosis and sex** using both **functional brain imaging data** (connectome matrices from fMRI scans) and **socio-demographic, emotional, and parenting-related metadata**.

This is a **multitask classification** problem with two binary targets:
- `ADHD_Outcome`: 0 = Other/None, 1 = ADHD
- `Sex_F`: 0 = Male, 1 = Female

### Real-World Significance  
Early and accurate diagnosis of ADHD in children and adolescents can lead to better outcomes through timely interventions. This competition explores whether functional brain imaging, combined with socio-emotional and parenting data, can be used to support such diagnoses. By advancing these models, we contribute to the growing field of computational psychiatry and help pave the way for more personalized and data-driven approaches to mental healthcare.

---

## 🔍 Data Exploration

### Dataset Description
This project uses neuroimaging and behavioral data from the Women in Data Science (WiDS) Kaggle competition to predict ADHD diagnosis. The dataset contains information from 1,200+ subjects organized into three main components:

**1. Target Variables:**
- Binary indicators for ADHD diagnosis and biological sex.

**Stored in `TRAINING_SOLUTIONS.csv`**

**2. Functional Brain Connectome Data:**
- High-dimensional representation of brain connectivity (19,900 features per subject).
- Features represent correlation strengths between pairs of brain regions.
- Column names follow a pattern like "0throw_1thcolumn", "0throw_2thcolumn", etc., indicating connections between specific brain regions.

**Training: `TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv`**

**Testing: `TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv`**

**3. Categorical Metadata:**
- Demographic information (enrollment year, study site).
- Child characteristics (race, ethnicity).
- MRI scan location.
- Parent education and occupation levels.

**Training: `TRAIN_CATEGORICAL_METADATA_new.csv`**

**Testing: `TEST_CATEGORICAL_METADATA.csv`**

**4. Quantitative Metadata:**
- Behavioral assessments (SDQ, APQ).
- Handedness scores (EHQ).
- Vision test results.
- Age at MRI scan.

**Training: `TRAIN_QUANTITATIVE_METADATA_new.csv`**

**Testing: `TEST_QUANTITATIVE_METADATA.csv`**


### Data Exploration and Preprocessing Approach
Our exploration and preprocessing pipeline consisted of the following key steps:

**1. Initial Data Assessment:**
- Loaded and examined the structure of all four datasets.
- Analyzed class distribution (found approximately equal representation of ADHD/non-ADHD participants).
- Checked for missing values and correlations between variables.
- Identified relationships between datasets using participant IDs.

**2. Functional Connectome Analysis:**
- Visualized connectivity matrices to understand brain region relationships.
- Analyzed the distribution of connectivity values across participants.
- Examined differences in connectivity patterns between ADHD and non-ADHD subjects.

**3. Behavioral and Socio-demographic Analysis:**
- Analyzed differences in behavioral metrics (SDQ, APQ) between groups.
- Identified the most important features for ADHD prediction.
- Explored correlations between behavioral measures and ADHD diagnosis.

**4. Data Preprocessing:**
- Handled missing values using appropriate imputation strategies.
- Applied feature scaling and normalization.
- Used Restricted Boltzmann Machines (RBMs) for feature extraction from connectome data.
- Prepared data for machine learning by splitting into training and testing sets.


### Exploratory Data Analysis Visualizations

**1. Feature Importance Analysis for ADHD Prediction**

![image](https://github.com/user-attachments/assets/52d103b8-8edf-4ad3-ae2a-a5859187f668)

This visualization shows the most important features for predicting ADHD in our model. Enrollment year and SDQ Composite score emerged as top predictors, followed by age at scan and SDQ generating impact. Notably, both demographic factors and behavioral metrics play significant roles in ADHD prediction, with SDQ metrics (measuring behavioral tendencies) appearing multiple times among the top features.

**2. Alternative Feature Importance Model**

![image](https://github.com/user-attachments/assets/353d61a9-f0bd-4d40-8aa4-751bcd707c3e)

In our alternative model, we see that the Edinburgh Handedness Questionnaire (EHQ) total score and age at scan are the most significant predictors. Parental involvement metrics from the Alabama Parenting Questionnaire (APQ_P_INV and APQ_P_ID) also show strong predictive power. This suggests that certain neurological indicators (like handedness) and parenting styles may correlate with ADHD diagnosis.
