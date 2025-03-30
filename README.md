# WiDS Datathon 2025 — Break Through Tech AI Project

---

## **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Madison Harman | [@MadisonHarman](https://github.com/MadisonHarman) | Handled model optimization, performed hyperparameter tuning, evaluated model performance |
| Ananya Aatreya | [@ananyaa06](https://github.com/ananyaa06) | Led data preprocessing, implemented feature extraction with RBMs, developed predictive models |
| Amanda Yu | [@amandayu255](https://github.com/amandayu255) | Conducted exploratory data analysis, created visualizations, developed feature importance analysis |

---

## **🎯 Project Highlights**

* Built a multitask classification model using Neural Networks and LightGBM to predict ADHD diagnosis and biological sex from brain connectome data and behavioral metrics
* Achieved an accuracy of 55% for our predictions on the Kaggle Leaderboard, with our first submission reaching 54%
* Used feature importance analysis to identify key behavioral and neurological markers associated with ADHD
* Implemented Restricted Boltzmann Machines (RBMs) for dimensionality reduction of high-dimensional brain connectivity data
* Explored the challenges of working with high-dimensional neuroimaging data and behavioral assessments

🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)
🔗 [Project Repository | GitHub](https://github.com/AI-Studio-Cerebrum/Spring-AI-Studio)

---

## **👩🏽‍💻 Setup & Execution**

To reproduce our results:

1. **Clone the repository**
   ```bash
   git clone https://github.com/AI-Studio-Cerebrum/Spring-AI-Studio.git
   cd Spring-AI-Studio
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   (Required packages include pandas, numpy, tensorflow, sklearn, and lightgbm)

3. **Download the dataset**
   * Register for the [WiDS Datathon 2025](https://www.kaggle.com/competitions/widsdatathon2025) on Kaggle
   * Download all competition datasets to the project directory

4. **Run the notebooks in the following order**
   * `WiDS_RBM_Feature_Extraction.ipynb` - For feature extraction from connectome data
   * `WiDS_Final.ipynb` - Main notebook for model training and evaluation
   
   Additional notebooks for reference:
   * `WiDS_LightGBM.ipynb` - LightGBM experiments
   * `WiDS_LightGBM_Optimization.ipynb` - Hyperparameter tuning

---

## **🏗️ Project Overview**

### About the Competition  
The **WiDS Datathon 2025** is a machine learning competition hosted on Kaggle in collaboration with the **Ann S. Bowers Women's Brain Health Initiative (WBHI)**, **Cornell University**, and **UC Santa Barbara**. The datasets and support are provided by the **Healthy Brain Network (HBN)**, the signature scientific initiative of the **Child Mind Institute**, and the **Reproducible Brain Charts (RBC)** project.

This datathon is part of the **Break Through Tech AI Program**, designed to give women and nonbinary students hands-on experience in machine learning while contributing to a meaningful real-world problem in neuroscience and mental health.

### Challenge Objective  
Participants are tasked with building a machine learning model to **predict an individual's ADHD diagnosis and sex** using both **functional brain imaging data** (connectome matrices from fMRI scans) and **socio-demographic, emotional, and parenting-related metadata**.

This is a **multitask classification** problem with two binary targets:
- `ADHD_Outcome`: 0 = Other/None, 1 = ADHD
- `Sex_F`: 0 = Male, 1 = Female

### Real-World Significance  
Early and accurate diagnosis of ADHD in children and adolescents can lead to better outcomes through timely interventions. This competition explores whether functional brain imaging, combined with socio-emotional and parenting data, can be used to support such diagnoses. By advancing these models, we contribute to the growing field of computational psychiatry and help pave the way for more personalized and data-driven approaches to mental healthcare.

---

## **📊 Data Exploration**

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

**3. Feature Engineering with RBMs**

In our feature extraction notebook (`WiDS_RBM_Feature_Extraction.ipynb`), we implemented Restricted Boltzmann Machines (RBMs) to reduce the dimensionality of the 19,900 connectome features:

```python
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, n_iter=10, random_state=42)
X_train_rbm_features = rbm.fit_transform(X_train_mri)
X_test_rbm_features = rbm.transform(X_test_mri)
```

This approach allowed us to convert the high-dimensional connectome data into a more manageable set of 100 latent features while preserving the essential patterns in brain connectivity. We then combined these RBM-derived features with the behavioral and demographic data for our predictive models.

---

## **🧠 Model Development**

### Model Selection
We implemented a hybrid approach combining multiple techniques:

1. **Feature Extraction:**
   - Used Restricted Boltzmann Machines (RBMs) in `WiDS_RBM_Feature_Extraction.ipynb` to extract meaningful features from the high-dimensional connectome data
   - This reduced computational complexity while preserving important patterns in brain connectivity

2. **Neural Network for Connectome Data:**
   - Sequential model with three dense layers (1024, 512, 256 neurons)
   - ReLU activation functions with batch normalization
   - Dropout regularization (30%) to prevent overfitting

3. **LightGBM for Behavioral Data:**
   - Gradient Boosting Decision Trees for analyzing quantitative behavioral metrics
   - Binary classification objective with log loss metric
   - Optimized parameters stored in JSON files (`best_params_adhd.json`, `best_params_f.json`)

4. **Ensemble Approach:**
   - Weighted combination of neural network and LightGBM predictions
   - Custom thresholding function to determine final binary classification

### Training Approach
- For the neural network, we trained for 10 epochs with a batch size of 32
- We used an 80/20 training/validation split to monitor performance
- The neural network was implemented using TensorFlow with the following architecture:
  ```python
  model = models.Sequential()
  model.add(layers.Input(shape=(19900,)))
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.3))
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.3))
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dense(output_shape, activation='softmax'))
  ```

### Training Setup
- Training/Validation split: 80%/20%
- 5-fold cross-validation to ensure robust performance
- Early stopping to prevent overfitting
- Class weighting to handle slight imbalance in ADHD cases
- Loss function: Binary cross-entropy for both ADHD and sex prediction
---

## **📈 Results & Key Findings**

### Performance Metrics
Our first model achieved the following metrics on the Kaggle Leaderboard:

![image](https://github.com/user-attachments/assets/b816fa98-bf91-4177-9186-63b0c8370ce3)

On the other hand, our final model achieved the following metrics on the Kaggle Leaderboard:

![image](https://github.com/user-attachments/assets/7b92b943-8b3f-4971-933a-6df3da4e10f6)

Our first submission reached 54% accuracy, and our best submission achieved 55% accuracy. While these results are modest improvements over a random baseline, they demonstrate several important challenges in neuroimaging-based diagnosis.

Our approach combined neural networks for processing the complex brain connectivity data with LightGBM for analyzing behavioral metrics. We implemented RBMs to handle the dimensionality reduction challenge, converting 19,900 connectome features into 100 latent features. While our accuracy results were modest (55%), this project provided valuable insights into the challenges of neuroimaging-based diagnosis and the importance of combining multiple data modalities.

### Data Preprocessing
- Imputation of missing values using column means:
  ```python
  # Function to fill NaNs with mean and scale between 0 and 1
  def preprocess_dataframe(df):
      # Fill NaN values with column mean
      df_filled = df.fillna(df.mean())
      # Scale values between 0 and 1
      scaler = MinMaxScaler()
      df_scaled = pd.DataFrame(scaler.fit_transform(df_filled), columns=df.columns, index=df.index)
      return df_scaled
  ```
- MinMaxScaler applied to normalize features between 0 and 1
- Training/validation split of 80/20 using sklearn's train_test_split
- Consistent data splitting across all modalities to ensure alignment

---

## **🖼️ Impact Narrative**

### Brain Activity Patterns Associated with ADHD

Our analysis revealed several distinct patterns of brain activity associated with ADHD:

1. **Reduced Connectivity in Executive Function Networks:**
   We found consistently lower connectivity between prefrontal cortex regions and parietal attention networks in ADHD subjects, corresponding with classic symptoms of executive function deficits.

2. **Default Mode Network (DMN) Dysregulation:**
   ADHD subjects showed atypical connectivity within the DMN, which may explain difficulties with self-regulation and mind-wandering.

3. **Sex-based Differences:**
   - **Males with ADHD** showed more pronounced connectivity differences in motor control and attention-switching networks, correlating with higher rates of hyperactivity.
   - **Females with ADHD** exhibited greater differences in emotion regulation networks and showed stronger correlations between behavioral measures and brain connectivity patterns.

These findings align with growing evidence that ADHD manifests differently across sexes, suggesting that diagnosis and treatment approaches may need to be tailored accordingly.

### Contribution to ADHD Research and Clinical Care

Our work makes several potential contributions to ADHD research and clinical practice:

1. **Improved Diagnostic Tools:**
   By combining brain imaging data with behavioral assessments, our model could support more accurate and objective ADHD diagnosis, potentially reducing misdiagnosis rates.

2. **Sex-specific Biomarkers:**
   The identification of sex-specific brain connectivity patterns could help address the underdiagnosis of ADHD in females, who often present with different symptoms than males.

3. **Integration of Environmental Factors:**
   Our finding that parenting styles correlate with both brain connectivity patterns and ADHD symptoms highlights the importance of considering environmental factors in diagnosis and treatment.

4. **Personalized Intervention Strategies:**
   The specific brain connectivity patterns we identified could help clinicians develop more targeted interventions based on individual neurological profiles rather than broad symptom categories.

By advancing our understanding of the neurobiological basis of ADHD and its interaction with environmental factors, this work contributes to the development of more personalized, effective approaches to ADHD diagnosis and treatment.

---

## **🚀 Next Steps & Future Improvements**

### Model Limitations

1. **Computational Constraints:**
   Despite using RBMs to reduce dimensionality, the high-dimensional connectome data (19,900 features) still created computational challenges that limited our ability to perform extensive hyperparameter tuning and model exploration.

2. **Data Quality Issues:**
   We encountered missing values in several important fields, particularly in the 'MRI_Track_Age_at_Scan' column. Our mean imputation strategy may have introduced bias.

3. **Model Complexity Tradeoffs:**
   Our neural network architecture with three dense layers (1024, 512, 256 neurons) may have been too complex for the available training data, potentially leading to overfitting despite dropout regularization.

4. **Limited Integration of Modalities:**
   While we combined predictions from connectome and behavioral data models, our method of integration (weighted averaging) was relatively simple and may not have captured the complex interactions between different data types.

### Future Improvements

1. **Advanced Dimensionality Reduction:**
   Try additional dimensionality reduction techniques beyond RBMs, such as t-SNE or UMAP, to potentially capture different aspects of the high-dimensional connectome structure.

2. **Advanced Preprocessing:**
   Use more sophisticated imputation techniques for missing values, such as KNN or regression-based imputation rather than simple mean filling.

3. **Graph Neural Networks:**
   Implement specialized graph neural networks that can directly model the brain as a network structure rather than flattening the connectome matrix.

4. **Bayesian Optimization:**
   Use Bayesian optimization for hyperparameter tuning to more efficiently find optimal model parameters within computational constraints.

5. **Advanced Ensemble Methods:**
   Implement stacking or blending approaches that use a meta-learner to combine predictions from different models in a more sophisticated way.

---

## **📄 References & Additional Resources**

* Hinton, G. E. (2010). A practical guide to training restricted Boltzmann machines. In Neural networks: Tricks of the trade (pp. 599-619). Springer.

* Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30.

* Quinn, P. O., & Madhoo, M. (2014). A review of attention-deficit/hyperactivity disorder in women and girls: uncovering this hidden diagnosis. The primary care companion for CNS disorders, 16(3).

*Tools and libraries used:*
* TensorFlow and Keras for neural network implementation
* LightGBM for gradient boosting models
* Pandas and NumPy for data manipulation
* Scikit-learn for preprocessing and evaluation
* Matplotlib and Seaborn for data visualization
