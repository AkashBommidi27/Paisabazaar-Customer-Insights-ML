Paisabazaar Customer Insights & Risk Management
Overview
This project leverages machine learning and financial analytics to derive actionable insights from customer and policy data at Paisabazaar, a leading financial services platform. The goal is to enhance customer engagement, optimize risk management, and drive business growth through data-driven strategies. Key tasks include customer segmentation, credit score prediction, loan risk assessment, churn risk detection, and identifying cross-selling opportunities.
The dataset includes customer demographics, financial behavior, and credit metrics, forming the foundation for advanced analytics and predictive modeling. The results provide strategic recommendations for personalized marketing, loan approvals, and churn prevention.



Key Details




Dataset
Anonymized customer data sourced from Google Drive


Notebook
Paisabazaar_Project.ipynb


Results
Saved in Result_df.csv


GitHub
AkashBommidi27/Paisabazaar-Customer-Insights-ML



Project Objectives

Understand Customer Behavior: Analyze preferences and financial patterns.
Segment Customers: Group customers for targeted marketing and personalized offers.
Predict Credit Scores: Develop a model to classify creditworthiness.
Assess Loan Risk: Identify high-risk borrowers to minimize defaults.
Detect Churn Risk: Flag customers at risk of disengagement or default.
Identify Cross-Selling Opportunities: Recommend relevant financial products based on customer profiles.


Dataset Description
The dataset contains comprehensive customer-level information, including:



Category
Features



Demographics
Age, Occupation, Annual Income, Monthly Inhand Salary


Financial Metrics
Number of Bank Accounts, Credit Cards, Loans, Outstanding Debt, Credit Utilization Ratio


Credit Behavior
Payment Delays, Credit Inquiries, Credit Mix, Payment Behavior


Target Variable
Credit Score (Good, Standard, Poor)


Key Features:

Total_EMI_per_month: Sum of monthly loan installments.
Outstanding_Debt: Total unpaid debt.
Monthly_Balance: Net monthly balance after expenses.
Delay_from_due_date: Average payment delay in days.
Amount_invested_monthly: Monthly investment amount.

For a detailed feature explanation, refer to the notebook.

Methodology
1. Data Preprocessing

Libraries: pandas, numpy for data manipulation; matplotlib, seaborn for visualization.
Steps:
Handled missing values and outliers.
Encoded categorical variables using LabelEncoder.
Standardized numerical features with StandardScaler.
Split data into training and testing sets using train_test_split.



2. Exploratory Data Analysis (EDA)

Analyzed distributions of key financial metrics.
Identified correlations between income, debt, and credit score.
Visualized payment behavior and credit utilization patterns.

3. Machine Learning Models

Credit Score Prediction:
Model: XGBClassifier with SMOTE for class imbalance.
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



Task
Key Metrics
Outcome



Credit Score Prediction
~84% accuracy (post-SHAP optimization)
Automated creditworthiness evaluation for faster loan approvals.


Customer Segmentation
4 clusters: Premium, Balanced, Emerging, At-Risk
Tailored marketing strategies for each segment.


Loan Risk Assessment
Flagged high-risk borrowers
Risk-adjusted loan terms to reduce non-performing assets (NPAs).


Churn Risk Prediction
Top 25% flagged as high-risk (Churn_Risk_Flag = 1)
Proactive retention strategies for at-risk customers.


Key Insights

Credit Score Predictors: Total_EMI_per_month, Outstanding_Debt, Monthly_Balance, Delay_from_due_date, Amount_invested_monthly.
Segmentation Profiles:
Premium: High-income, disciplined borrowers.
Balanced: Stable earners with moderate activity.
Emerging: Low-income, consistent customers.
At-Risk: Financially vulnerable with high debt or delays.


Churn Overlap: Strong correlation between At-Risk and low-end Balanced segments.


Business Strategies
The project delivers actionable strategies to align with Paisabazaar’s mission:



Segment
Recommended Products
Strategy



Premium
Mutual funds, premium credit cards, term loans
Upsell high-value financial services


Balanced
Insurance, low-risk investments
Cross-promote smart bundling offers


Emerging
Micro-loans, budgeting tools
Promote savings and discipline tools


At-Risk
Credit repair, advisory plans
Offer financial literacy programs


Business Impact

Personalized Marketing: Tailored recommendations boost conversion and loyalty.
Risk Mitigation: Automated credit scoring reduces defaults and enhances portfolio health.
Customer Retention: Proactive churn flagging minimizes drop-off.
Scalable Growth: Data-driven insights support inclusive financial services at scale.


Tools & Technologies



Category
Tools



Programming Language
Python


Libraries
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, imbalanced-learn


Environment
Jupyter Notebook (Google Colab with GPU support)


Data Source
Google Drive (via gdown)



How to Run the Project

Clone the Repository:
git clone https://github.com/AkashBommidi27/Paisabazaar-Customer-Insights-ML.git


Install Dependencies:
pip install pandas numpy scikit-learn xgboost seaborn matplotlib gdown imbalanced-learn


Download the Dataset:

The notebook fetches the dataset using gdown.
Alternatively, place paisabazaar_dataset.csv in the project directory.


Run the Notebook:

Open Paisabazaar_Project.ipynb in Jupyter Notebook or Google Colab.
Execute cells sequentially to reproduce analysis and results.


View Results:

Outputs saved in Result_df.csv.
Visualizations and metrics displayed in the notebook.




Future Enhancements

Advanced Models: Explore deep learning (e.g., neural networks) for improved prediction.
Real-Time Analytics: Deploy models as APIs for live customer scoring.
Feature Engineering: Incorporate external data (e.g., market trends) for richer insights.
Explainability: Enhance SHAP-based feature importance for customer-facing transparency.


Contact
For questions or collaboration opportunities, reach out to:



Platform
Details



GitHub
AkashBommidi27


Email
[Your Email Address] (please add your email here)



Thank you for exploring this project! Contributions and feedback are welcome.
