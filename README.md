Loan Approval Prediction using Machine Learning
Problem Statement

Loan approval is a critical decision-making process for financial institutions. Approving loans for high-risk customers can lead to financial losses, while rejecting eligible customers may result in lost business opportunities.

This project aims to develop multiple machine learning classification models to predict loan approval status using customer financial, credit, and demographic information. The goal is to compare different classification techniques and identify the best performing model based on evaluation metrics.

Dataset Description

The dataset contains customer financial and loan-related details used to predict loan approval status.

Dataset Details

Total Records: 50,000

Total Features: 20 Original Features

Features After Encoding: 24

Target Variable: Loan Status

Target Variable

1 → Loan Approved / Safe Borrower

0 → Loan Rejected / Risky Borrower

Feature Description
Feature	Description
age	Customer age
occupation_status	Employment status
years_employed	Years of employment
annual_income	Annual income of customer
credit_score	Creditworthiness score
credit_history_years	Length of credit history
savings_assets	Customer savings and assets
current_debt	Existing debt amount
defaults_on_file	Past default records
delinquencies_last_2yrs	Late payments in last two years
derogatory_marks	Negative credit history marks
product_type	Type of loan product
loan_intent	Purpose of loan
loan_amount	Loan amount requested
interest_rate	Interest rate of loan
debt_to_income_ratio	Debt compared to income
loan_to_income_ratio	Loan compared to income
payment_to_income_ratio	Payment compared to income
Data Preprocessing

The following preprocessing steps were applied:

Removed customer ID column

Converted categorical features using One-Hot Encoding

Split dataset into training and testing sets (80% training, 20% testing)

Applied feature scaling using StandardScaler for scale-sensitive models

Models Used and Evaluation Metrics

Six machine learning classification models were implemented:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbor (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble Model)

XGBoost (Ensemble Model)

Model Performance Comparison
ML Model Name	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.8653	0.9446	0.8711	0.8865	0.8787	0.7274
Decision Tree	0.8715	0.8696	0.8794	0.8885	0.8839	0.7401
KNN	0.8673	0.9320	0.8542	0.9152	0.8836	0.7320
Naive Bayes	0.7436	0.8280	0.7992	0.7135	0.7539	0.4915
Random Forest (Ensemble)	0.9099	0.9725	0.9152	0.9217	0.9185	0.8178
XGBoost (Ensemble)	0.9243	0.9826	0.9235	0.9404	0.9319	0.8469
Observations on Model Performance
Logistic Regression

Performed well with high AUC score and balanced classification performance. However, it assumes linear relationships between features and target variable.

Decision Tree

Provides interpretable results and balanced performance but may overfit the dataset.

K-Nearest Neighbor

Achieved high recall, indicating strong ability to identify safe borrowers. Performance depends on feature scaling and selection of neighbors.

Naive Bayes

Fast and computationally efficient but showed lower performance due to independence assumptions among features.

Random Forest (Ensemble)

Improved performance by combining multiple decision trees, reducing overfitting and improving accuracy.

XGBoost (Ensemble)

Achieved the best overall performance across all evaluation metrics. The boosting algorithm improves predictions by learning from previous errors.

Best Performing Model

XGBoost achieved the highest performance with:

Accuracy: 92.43%

AUC Score: 0.9826

F1 Score: 0.9319

MCC Score: 0.8469

This indicates strong reliability in predicting loan approval outcomes.
