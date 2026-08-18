# College Sleep and GPA Prediction

## 1. Problem Statement

The objective of this project is to develop a machine learning classification system using the College Sleep and GPA dataset.

The target variable is `under_6h_sleep`, which represents whether a student belongs to the category of students sleeping fewer than six hours.

The project compares five machine learning classification algorithms:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest

The models are evaluated using the following performance metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The final objective is to select a suitable classification model and deploy the prediction system using Streamlit.

---

## 2. Dataset Description

The selected dataset is:

`college_sleep_and_gpa.csv`

The dataset contains information related to college students' sleep patterns and academic characteristics.

For this project, the target variable is:

`under_6h_sleep`

The prepared modeling dataset contains:

- Number of observations: 634
- Number of input features: 18
- Training observations: 507
- Testing observations: 127
- Test size: 20%
- Random state: 42

The dataset was divided into training and testing sets using a stratified train-test split to maintain the target-class distribution.

### Data Preprocessing

The machine learning workflow includes preprocessing of numerical and categorical features.

Numerical features are handled using:

- Missing-value imputation using the median
- Standard scaling

Categorical features are handled using:

- Missing-value imputation using the most frequent value
- One-hot encoding
- Unknown categories are handled using `handle_unknown="ignore"`

The preprocessing steps are integrated with the machine learning models using a Scikit-learn pipeline.

---

## 3. GitHub Repository

The complete project repository is available at:

**GitHub Repository:**  
https://github.com/Shruthi0213/college-sleep-gpa-project

The repository contains the trained machine learning models, preprocessing component, test dataset, project dependencies, documentation, and Streamlit application files.

---

## 4. Machine Learning Models

Five classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest

---

## 5. Model Evaluation

The models were evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 0.9370 | 0.9966 | 1.0000 | 0.6923 | 0.8182 | 0.8009 |
| Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## 6. Model-wise Observations

### 6.1 Logistic Regression

Logistic Regression achieved perfect performance on the selected test dataset.

The model obtained:

- Accuracy: 1.0000
- AUC: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1 Score: 1.0000
- MCC: 1.0000

The results indicate that Logistic Regression correctly classified all observations in the test set and achieved perfect discrimination between the two target classes on this evaluation split.

---

### 6.2 Decision Tree

The Decision Tree also achieved perfect performance across all evaluation metrics.

The model obtained:

- Accuracy: 1.0000
- AUC: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1 Score: 1.0000
- MCC: 1.0000

The model correctly classified all observations in the selected test dataset.

---

### 6.3 K-Nearest Neighbors (KNN)

KNN achieved an Accuracy of 0.9370 and an AUC of 0.9966.

The model achieved a Precision of 1.0000, indicating that its positive predictions were correct. However, its Recall was 0.6923, which indicates that some positive cases were not identified.

Consequently, the KNN model obtained:

- F1 Score: 0.8182
- MCC: 0.8009

Among the five evaluated models, KNN produced the lowest overall performance on the selected test dataset, particularly in Recall, F1 Score, and MCC.

---

### 6.4 Naive Bayes

Naive Bayes achieved perfect performance across all six evaluation metrics.

The model obtained:

- Accuracy: 1.0000
- AUC: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1 Score: 1.0000
- MCC: 1.0000

The results indicate perfect classification performance on the selected test split.

---

### 6.5 Random Forest

Random Forest achieved perfect performance across all six evaluation metrics.

The model obtained:

- Accuracy: 1.0000
- AUC: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1 Score: 1.0000
- MCC: 1.0000

The results indicate that Random Forest correctly classified all observations in the selected test dataset.

---

## 7. Overall Model Comparison

Four of the five evaluated models achieved identical perfect scores across all six evaluation metrics:

- Logistic Regression
- Decision Tree
- Naive Bayes
- Random Forest

KNN achieved a lower performance, mainly because of its lower Recall, F1 Score, and MCC.

Therefore, there is a four-way tie for the highest observed performance on the selected test split.

### Selected Model: Logistic Regression

Logistic Regression is selected as the project model because it achieved perfect performance on the test dataset while providing a relatively simple and interpretable classification approach.

Its evaluation results are:

| Metric | Score |
|---|---:|
| Accuracy | 1.0000 |
| AUC | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |
| MCC | 1.0000 |

The selection should be interpreted as a project-level model choice based on the current evaluation split. The identical perfect scores achieved by several models indicate that additional validation could be useful before making claims about performance on unseen real-world data.

---

## 8. Project Structure

The project repository is organized as follows:

```text
college-sleep-gpa-project/
│
├── app.py
├── README.md
├── requirements.txt
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── preprocessor.pkl