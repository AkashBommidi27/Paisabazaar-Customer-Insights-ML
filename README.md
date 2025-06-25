Paisabazaar Customer Insights & Risk Management
Project Overview
This project focuses on leveraging machine learning and financial analytics to derive actionable insights from customer and policy data at Paisabazaar, a leading financial services platform. The objective is to enhance customer engagement, optimize risk management, and drive business growth through data-driven strategies. Key tasks include customer segmentation, credit score prediction, loan risk assessment, churn risk detection, and identifying cross-selling opportunities.
The dataset, comprising customer demographics, financial behavior, and credit metrics, serves as the foundation for advanced analytics and predictive modeling. The results provide strategic recommendations for personalized marketing, loan approvals, and churn prevention.
🔗 Dataset: Sourced from a Google Drive link (anonymized customer data).🔗 Notebook: Paisabazaar_Project.ipynb🔗 Results: Saved in Result_df.csv

Project Objectives

Customer Behavior Analysis: Understand customer preferences and financial patterns.
Customer Segmentation: Group customers for targeted marketing and personalized offers.
Credit Score Prediction: Develop a model to classify customers' creditworthiness.
Loan Risk Assessment: Identify high-risk borrowers to minimize defaults.
Churn Risk Detection: Flag customers at risk of disengagement or default.
Cross-Selling Opportunities: Recommend relevant financial products based on customer profiles.


Dataset Description
The dataset contains rich customer-level information, including:

Demographics: Age, occupation, annual income, monthly in-hand salary.
Financial Metrics: Number of bank accounts, credit cards, loans, outstanding debt, credit utilization ratio.
Credit Behavior: Payment delays, credit inquiries, credit mix, payment behavior.
Target Variable: Credit score (Good, Standard, Poor).

Key Features:

Total_EMI_per_month: Sum of monthly loan installments.
Outstanding_Debt: Total unpaid debt.
Monthly_Balance: Net monthly balance after expenses.
Delay_from_due_date: Average payment delay in days.
Amount_invested_monthly: Monthly investment amount.

For a detailed feature explanation, refer to the notebook.

Methodology
1. Data Preprocessing

Libraries Used: Pandas, NumPy for data manipulation; Matplotlib, Seaborn for visualization.
Preprocessing Steps:
Handled missing values and outliers.
Encoded categorical variables using LabelEncoder.
Standardized numerical features with StandardScaler.
Split data into training and testing sets using train_test_split.



2. Exploratory Data Analysis (EDA)

Analyzed distributions of key financial metrics.
Identified correlations between features like income, debt, and credit score.
Visualized payment behavior and credit utilization patterns.

3. Machine Learning Models

Credit Score Prediction:
Model: XGBClassifier with SMOTE for handling class imbalance.
Feature Selection: Optimized using SHAP values.
Evaluation Metrics: Accuracy, classification report, confusion matrix.


Customer Segmentation:
Model: KMeans clustering (4 clusters).
Features: Income, EMI, debt, disposable income.
Labels: Premium, Balanced, Emerging, At-Risk.



4. Custom Scoring

Loan Risk Score: Combined payment delays, debt, and credit utilization.
Churn Risk Score: Weighted score based on delay metrics and cash flow.


Results
1. Credit Score Prediction

Accuracy: ~84% (post-SHAP optimization).
Key Predictors: Total_EMI_per_month, Outstanding_Debt, Monthly_Balance, Delay_from_due_date, Amount_invested_monthly.
Impact: Enables automated credit evaluations and faster loan approvals.

2. Customer Segmentation

Clusters:
Premium: High-income, disciplined borrowers.
Balanced: Stable earners with moderate financial activity.
Emerging: Low-income but consistent customers.
At-Risk: Financially vulnerable with high debt or delays.


Strategic Use: Tailored marketing campaigns for each segment.

3. Loan Risk Assessment

Identified high-risk borrowers based on repayment history and debt levels.
Outcome: Supports risk-adjusted loan terms to reduce non-performing assets (NP 2).

4. Churn Risk Prediction

Flagged top 25% of customers as high-risk (Churn_Risk_Flag = 1).
Overlap: Strong correlation with At-Risk and low-end Balanced segments.
Outcome: Proactive retention strategies for at-risk customers.


Business Strategies
This project delivers actionable strategies aligned with Paisabazaar’s goals:



Segment
Recommended Products
Strategy



Premium
Mutual funds, premium credit cards, term loans
Upsell high-value financial services


Balanced
Insurance, low-risk investments funds
Cross-promote smart bundling offers


Emerging
Micro-loans, budgeting tools
Promote savings and discipline tools


At-Risk
Credit repair, advisory plans
Offer financial literacy programs


Key Impacts

Personalized Marketing: Tailored product recommendations improve conversion rates and customer loyalty.
Risk Mitigation: Automated credit scoring reduces defaults and enhances portfolio health.
Customer Retention: Proactive churn flagging flagging minimizes customer drop-off.
Scalable Growth: Data-driven insights strategies support Paisabazaar’s mission of inclusive financial services.


Tools & Technologies

Programming Language: Python
Libraries:
Data Manipulation: pandas, numpy
Visualization: matplotlib, seaborn
Machine Learning: scikit-learn, xgboost
Preprocessing: sklearn.preprocessing, sklearn.preprocessing, imbalanced-learn


Environment: Jupyter Notebook (Google Colab with Colab)
Data Source: Google Drive (via gdown)


How to Run the Project

Clone the Repository:
git clone https://github.com/AkashBommidi27/Paisabazaar-Customer-Insights-ML.git


Install Dependencies:
pip install pandas numpy scikit-learn xgboost seaborn matplotlib gdown imbalanced-learn


Download the Dataset:

The notebook automatically fetches the dataset from Google Drive using gdown.
Alternatively, place paisabazaar_dataset.csv in the project directory.


Run the Notebook:

Open Paisabazaar_Project.ipynb in Jupyter Notebook or Google Colab.
Execute cells sequentially to reproduce the analysis and results.


View Results:

Outputs are saved in Result_df.csv.
Visualizations and model metrics are displayed in the notebook.




Future Enhancements

Advanced Models: Experiment with deep learning (e.g., neural networks) for credit score prediction.
Real-Time Analytics: Deploy models as APIs for live customer scoring.
Feature Engineering: Incorporate external data (e.g., market trends) for richer insights.
Explainability: Enhance SHAP-based feature importance for customer-facing transparency.


Contact
For questions or collaboration opportunities, reach out to:

GitHub: AkashBommidi27
Email: [Your Email Address] (please add your email here)


Thank you for exploring this project! Your feedback and contributions are welcome.
