# Paisabazaar Customer Insights and Risk Management

## Overview

This project utilizes machine learning and financial analytics to derive actionable insights from customer and policy data at Paisabazaar, a leading financial services platform. The primary goal is to enhance customer engagement, optimize risk management, and drive business growth through data-driven strategies. The key tasks include customer segmentation, credit score prediction, loan risk assessment, churn risk detection, and identifying cross-selling opportunities.

The dataset comprises customer demographics, financial behavior, and credit metrics, serving as the foundation for advanced analytics and predictive modeling. The results provide strategic recommendations for personalized marketing, loan approvals, and churn prevention.

Key Details:

| Item | Description |
|------|-------------|
| Dataset | [Anonymized customer data sourced from Google Drive](https://drive.google.com/file/d/1tpaMVcgegVvm5_zJUyWnct3DlG7EtDo7/view?usp=sharing) |
| Notebook | [Paisabazaar_Project.ipynb](Paisabazaar_Project.ipynb) |
| Results | Saved in [Result_df.csv](Result_df.csv) |
| GitHub | [AkashBommidi27/Paisabazaar-Customer-Insights-ML](https://github.com/AkashBommidi27/Paisabazaar-Customer-Insights-ML) |

## Project Objectives

- Understand customer behavior and preferences through data analysis.
- Segment customers into meaningful groups for targeted marketing.
- Predict credit scores to assess customer creditworthiness.
- Assess loan default risks to minimize financial losses.
- Detect customers at risk of churn or disengagement.
- Identify opportunities for cross-selling financial products.

## Dataset Description

The dataset contains comprehensive customer-level information, including:

| Category | Features |
|----------|----------|
| Demographics | Age, Occupation, Annual Income, Monthly Inhand Salary |
| Financial Metrics | Number of Bank Accounts, Credit Cards, Loans, Outstanding Debt, Credit Utilization Ratio |
| Credit Behavior | Payment Delays, Credit Inquiries, Credit Mix, Payment Behavior |
| Target Variable | Credit Score (Good, Standard, Poor) |

Key Features:
- Total_EMI_per_month: Sum of monthly loan installments.
- Outstanding_Debt: Total unpaid debt.
- Monthly_Balance: Net monthly balance after expenses.
- Delay_from_due_date: Average payment delay in days.
- Amount_invested_monthly: Monthly investment amount.

For a detailed feature explanation, refer to the [notebook](Paisabazaar_Project.ipynb).

## Methodology

### Data Preprocessing
- Libraries: pandas, numpy for data manipulation; matplotlib, seaborn for visualization.
- Steps:
  - Handled missing values and outliers.
  - Encoded categorical variables using LabelEncoder.
  - Standardized numerical features with StandardScaler.
  - Split data into training and testing sets using train_test_split.

### Exploratory Data Analysis (EDA)
- Analyzed distributions of key financial metrics.
- Identified correlations between income, debt, and credit score.
- Visualized payment behavior and credit utilization patterns.

### Machine Learning Models
- Credit Score Prediction:
  - Model: XGBClassifier with SMOTE for class imbalance.
  - Feature Selection: Optimized using SHAP values.
  - Evaluation Metrics: Accuracy, classification report, confusion matrix.
- Customer Segmentation:
  - Model: KMeans clustering (4 clusters).
  - Features: Income, EMI, debt, disposable income.
  - Labels: Premium, Balanced, Emerging, At-Risk.

### Custom Scoring
- Loan Risk Score: Combined payment delays, debt, and credit utilization.
- Churn Risk Score: Weighted score based on delay metrics and cash flow.

## Results

| Task | Key Metrics | Outcome |
|------|-------------|---------|
| Credit Score Prediction | ~84% accuracy (post-SHAP optimization) | Automated creditworthiness evaluation for faster loan approvals. |
| Customer Segmentation | 4 clusters: Premium, Balanced, Emerging, At-Risk | Tailored marketing strategies for each segment. |
| Loan Risk Assessment | Flagged high-risk borrowers | Risk-adjusted loan terms to reduce non-performing assets. |
| Churn Risk Prediction | Top 10% flagged as high-risk (Churn_Risk_Flag = 1) | Proactive retention strategies for at-risk customers. |

Key Insights:
- Credit Score Predictors: Total_EMI_per_month, Outstanding_Debt, Monthly_Balance, Delay_from_due_date, Amount_invested_monthly.
- Segmentation Profiles:
  - Premium: High-income, disciplined borrowers.
  - Balanced: Stable earners with moderate activity.
  - Emerging: Low-income, consistent customers.
  - At-Risk: Financially vulnerable with high debt or delays.
- Churn Overlap: Strong correlation between At-Risk and low-end Balanced segments.

## Business Strategies

The project delivers actionable strategies aligned with Paisabazaar’s mission:

| Segment | Recommended Products | Strategy |
|---------|----------------------|----------|
| Premium | Mutual funds, premium credit cards, term loans | Upsell high-value financial services |
| Balanced | Insurance, low-risk investments | Cross-promote smart bundling offers |
| Emerging | Micro-loans, budgeting tools | Promote savings and discipline tools |
| At-Risk | Credit repair, advisory plans | Offer financial literacy programs |

Business Impact:
- Personalized Marketing: Tailored recommendations improve conversion and loyalty.
- Risk Mitigation: Automated credit scoring reduces defaults and enhances portfolio health.
- Customer Retention: Proactive churn flagging minimizes drop-off.
- Scalable Growth: Data-driven insights support inclusive financial services at scale.

## Tools and Technologies

| Category | Tools |
|----------|-------|
| Programming Language | Python |
| Libraries | pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, imbalanced-learn |
| Environment | Jupyter Notebook (Google Colab with GPU support) |
| Data Source | Google Drive (via gdown) |

## How to Run the Project

1. Clone the Repository:
   ```bash
   git clone https://github.com/AkashBommidi27/Paisabazaar-Customer-Insights-ML.git
   ```

2. Install Dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost seaborn matplotlib gdown imbalanced-learn
   ```

3. Download the Dataset:
   - The notebook fetches the dataset using gdown.
   - Alternatively, place paisabazaar_dataset.csv in the project directory.

4. Run the Notebook:
   - Open Paisabazaar_Project.ipynb in Jupyter Notebook or Google Colab.
   - Execute cells sequentially to reproduce analysis and results.

5. View Results:
   - Outputs saved in Result_df.csv.
   - Visualizations and metrics displayed in the notebook.

## Future Enhancements

- Explore advanced models like neural networks for improved prediction.
- Deploy models as APIs for real-time customer scoring.
- Incorporate external data (e.g., market trends) for richer insights.
- Enhance SHAP-based feature importance for customer-facing transparency.

## Contact

For questions or collaboration opportunities, reach out to:

| Platform | Details |
|----------|---------|
| GitHub | [AkashBommidi27](https://github.com/AkashBommidi27) |
| Email | [Your Email Address] |

Thank you for exploring this project. Contributions and feedback are welcome.
