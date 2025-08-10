NOTE: Dataset not available due to file size being huge!

# 📈 Credit Score Prediction for Enhanced Financial Risk Management

This project focuses on developing a robust machine learning model to accurately predict individual credit scores, categorizing them as 'Poor', 'Standard', or 'Good'. This capability is crucial for financial institutions like Paisabazaar, enabling them to make more informed lending decisions, mitigate financial risks, and offer personalized services.

---

## 🎯 Problem Statement

Paisabazaar, a prominent financial services company, faces the challenge of efficiently and precisely assessing customer creditworthiness. Inaccurate credit score classifications can lead to sub-optimal loan approvals, increased rates of loan defaults, and an inability to provide tailored financial product recommendations. This directly impacts the company's operational efficiency, risk management capabilities, and overall profitability. The core problem addressed is to build a reliable predictive model that overcomes these challenges by classifying credit scores based on diverse customer data.

---

## 📊 Dataset

The project utilizes a dataset containing various financial and demographic attributes of individuals, along with their associated credit scores. This rich dataset provides the foundation for training and evaluating the predictive models.

---

## 🚀 Project Phases & Methodology

The project followed a comprehensive machine learning workflow:

### 1. Data Preprocessing & Feature Engineering

* **Data Cleaning:** Handled missing values through imputation (e.g., median for numerical features).
* **Outlier Treatment:** Applied IQR-based capping to manage extreme values, ensuring model stability.
* **Categorical Feature Encoding:**
    * `Credit_History_Age`: Converted from 'X years Y months' to total numerical months.
    * `Type_of_Loan`: Transformed into multiple binary columns to capture multi-label information.
    * Ordinal features (`Credit_Mix`, `Payment_of_Min_Amount`) and the target `Credit_Score` were mapped to numerical representations.
    * Nominal features (`Occupation`, `Payment_Behaviour`) were one-hot encoded.
* **Feature Manipulation:** Identified and addressed highly correlated features (e.g., `Annual_Income` and `Monthly_Inhand_Salary`) to reduce redundancy.
* **New Feature Creation:** Engineered insightful features like `Debt_to_Income_Ratio`, `EMI_to_Salary_Ratio`, and `Payment_Consistency` to capture deeper financial behavior.
* **Data Transformation:** Applied `np.log1p` to skewed numerical features to normalize their distributions.
* **Feature Scaling:** Utilized `StandardScaler` to bring all numerical features to a common scale (mean 0, std dev 1), crucial for distance-based algorithms and faster convergence.
* **Dimensionality Reduction (Conditional):** Employed `PCA` (Principal Component Analysis) to reduce the number of features while retaining 95% of the variance, applied if the feature count exceeded a certain threshold after encoding.

### 2. Imbalance Handling

* **Stratified Data Splitting:** The dataset was split into training (80%) and testing (20%) sets using `stratify=y` to ensure that the proportions of 'Poor', 'Standard', and 'Good' credit scores were preserved in both sets, vital for reliable evaluation.
* **SMOTE (Synthetic Minority Over-sampling Technique):** Applied to the training data to synthesize new examples for the minority classes (`Poor` and `Good`). This balanced the dataset, preventing models from being biased towards the majority 'Standard' class and ensuring fair learning.

### 3. Model Development & Optimization

* **Algorithms Implemented:**
    * **Logistic Regression:** A robust linear model, serving as a strong baseline.
    * **Decision Tree Classifier:** Capable of capturing non-linear relationships.
    * **Random Forest Classifier:** An ensemble method, building multiple decision trees for enhanced robustness and accuracy.
* **Hyperparameter Optimization:** `RandomizedSearchCV` was used with 5-fold cross-validation for each model. This technique efficiently explored a wide range of hyperparameter combinations, identifying optimal configurations to maximize model performance (specifically, `f1_weighted` score).

### 4. Model Evaluation & Selection

Models were rigorously evaluated using:
* **Accuracy:** Overall correct predictions.
* **Precision, Recall, F1-score:** Detailed class-specific performance, crucial for understanding false positives (e.g., misclassifying 'Good' as 'Poor') and false negatives (e.g., missing actual 'Poor' cases).
* **Confusion Matrix:** Visualizing specific misclassification patterns.
* **ROC AUC Curves (One-vs-Rest):** Assessing the model's ability to distinguish between classes across all possible thresholds.

---

## ✨ Results & Business Impact

The **Optimized Random Forest Classifier** emerged as the top-performing model. It consistently delivered the highest overall accuracy and weighted F1-score. Critically, it demonstrated:

* **Exceptional Recall for 'Poor' Credit Scores (0.86):** Significantly reducing the risk of loan defaults by identifying most high-risk customers.
* **Strong Precision for 'Good' Credit Scores (0.78):** Ensuring reliable identification of genuinely creditworthy customers for profitable offerings.

This model will empower Paisabazaar to:
* **Automate & Standardize Credit Assessment:** Reducing manual effort and human bias.
* **Mitigate Financial Risks:** By accurately flagging high-risk applicants.
* **Optimize Lending Strategies:** Leading to more confident loan approvals and personalized product recommendations.
* **Boost Profitability & Efficiency:** Through reduced losses and streamlined operations.

---

## 🛠️ Technologies Used

* **Python:** Programming Language
* **Pandas:** Data Manipulation & Analysis
* **NumPy:** Numerical Operations
* **Scikit-learn:** Machine Learning Algorithms & Preprocessing
* **Imbalanced-learn (imbalanced-learn):** Handling Imbalanced Datasets (SMOTE)
* **Matplotlib:** Plotting & Visualization
* **Seaborn:** Statistical Data Visualization
* **SciPy:** Statistical Distributions (for RandomizedSearchCV)

```
