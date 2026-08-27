# B2B SaaS Churn Analytics
### Predicting revenue at risk before it walks out the door

Churn is not a sudden rage-quit. It is a slow, measurable fade. This project builds a full pipeline that spots that fade, explains why it happens, and flags active accounts before they cancel.

**[Live Prediction App](ml_deployment/)** · **[Dashboard Screenshots](#dashboard)** · **[SQL Queries](data_prep/)** · **[ML Notebook](MachineLearning.ipynb)**

![SQL](https://img.shields.io/badge/SQL-MySQL-blue) ![Python](https://img.shields.io/badge/Python-Pandas%20%7C%20scikit--learn-yellow) ![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange) ![Streamlit](https://img.shields.io/badge/App-Streamlit-red) ![PowerBI](https://img.shields.io/badge/Dashboard-Power%20BI-black)

---

## The Bottom Line

> By modeling the historical "pre-churn signature," **241 active accounts** are flagged as high-risk today. Together they represent **$521,999 in Monthly Recurring Revenue** on the chopping block.

  | Historical Snapshot (Power BI) | Value |
  |---|---|
  | MRR Churn Rate | 7.74% |
  | Churned ARR (Total) | $641.88K |
  | Logo Churn Rate | 14.57% |
  | High-Risk Ticket Ratio | 34.75% |
  
  | Predictive Model Output | Value |
  |---|---|
  | Active accounts flagged high-risk | 241 |
  | MRR currently at risk | $521,999 |

*The dashboard tells you what already happened. The model tells you what's about to.*

---

## Who Is Churning? (Firmographics)

- **The Tier Effect**: Starter tier churn sits at **20.1%**, about **5x higher** than Enterprise (4.0%).
- **The Billing Lever**: Moving from monthly to annual billing cuts churn by **~50% (Starter)** and **~80% (Professional)**.
- **Industry Mismatch**:

  | High Churn (>20%) | Low Churn (<9%) |
  |---|---|
  | Real Estate | Hospitality |
  | Energy | Non-Profit |
  | Retail | Finance |
  | Education | Consulting |
  | Manufacturing | |

- **The "Paid" Anomaly**: Paid-channel Professional accounts churn at **26.5%**, almost **2x** the Paid Starter rate. This shows a higher tier does not protect against a poor channel fit.
- **The Cost of Scaling**: Peak sign-up volume in Q3 to Q4 2023 lines up with the highest churn cohorts on record (19% to 21%). This suggests volume was pushed over lead quality at that time.

---

## Why Are They Leaving? (Behavioral & Support Drivers)

| Signal | Retained Accounts | Churned Accounts |
|---|---|---|
| Avg. users per account | 13 | 7 |
| Avg. monthly actions | 1,400+ | ~110 |
| Champion's share of account activity | 22% | 33% |
| Avg. high/critical support tickets | 1.74 | 4.5 |

- **The Habit Gap**: Retention lines up with team-wide adoption, not use by one person. Churned accounts stay stuck around a single power user.
- **The Single Point of Failure**: When one "champion" drives a third of all activity, that person leaving often takes the account with them.
- **Usability Is Not the Problem**: When churned users do log in, they spend the same ~30 minutes in the app as top customers. The product works fine. The problem is how often they log in, not the experience itself.
- **Severity Beats Volume**: Filing a ticket does not predict churn, since 100% of customers do it at some point. Ticket severity does.

---

## The Pre-Churn Signature (Early Warning Signs)

- **The 60-Day Cliff**: Monthly activity drops **~65%** in the last two months before cancellation. This decay curve is easy to spot.
- **CSAT Sensitivity**: Satisfaction score is one of the strongest predictors in the dataset.

  | CSAT Score | Churn Rate |
  |---|---|
  | 1 | 57.7% |
  | 5 | 6.2% |

---

## Financial Impact

- **The Predictive Threat**: 241 active accounts show the same warning signs as past churners today, putting **$521K+ in MRR** at risk.
- **The Enterprise Trap**: Enterprise churn is rare (3 accounts on record) but costly. Each loss costs about **$35K in contract value** and **$5.8K in MRR**, which is why these accounts deserve extra retention effort.

---

## Strategic Recommendations

1. **Deploy an automated risk workflow**: Customer Success should reach out to the 241 flagged accounts using the "usage decay plus low CSAT" trigger.
2. **Incentivize annual contracts**: Update checkout and discounting to push annual billing for Starter and Professional plans as a main way to keep customers.
3. **Audit the Paid channel**: Review targeting and ad copy for Professional-tier paid campaigns to stop bringing in users who churn fast and do not fit well.
4. **Shift onboarding KPIs**: Move the focus from "feature education" to "team seat expansion," since getting many users into the habit is the best protection against champion turnover.
5. **Protect high-value tiers**: Give Enterprise accounts dedicated, VIP-level Customer Success check-ins, since a low churn rate hides a high churn cost.

---

## How It Was Built

| Stage | Tool | What Happened |
|---|---|---|
| Data Generation | Python (Faker) | Synthetic dataset: 350 companies, 18 months of history |
| Cleaning & EDA | SQL (MySQL, Docker) | 17 structured insights, NULL-handling strategy, staging → master table |
| Modeling | Python (scikit-learn) | Churn classification model |
| Explainability | SHAP | Feature-level explanations for individual predictions |
| Live Predictions | Streamlit | Interactive app for scoring any account |
| Executive Reporting | Power BI | Single-page dashboard covering overview and risk detail |

---

## Screenshots <a name="dashboard"></a>

**Executive Power BI Dashboard**
<img width="1093" height="973" alt="B2B_SaaS_Churn_Dashboard_Screenshot" src="https://github.com/user-attachments/assets/17bf8bad-7f66-4d7b-9f3d-3cbcc32025f5" />

**Live Streamlit Prediction App**
<!-- ![Streamlit App](images/streamlit_app.png) -->

**Shap Summary Plot**
<img width="758" height="405" alt="image" src="https://github.com/user-attachments/assets/b9da0d2b-7eb5-41b6-ab6d-a86243896c13" />


---

## Repository Structure

```
b2b-churn-analysis-with-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── /data
│     ├── company_profiles.xlsx
│     ├── subscriptions.xlsx
│     ├── support_tickets.xlsx
│     ├── usage_events.xlsx
│     └── customer_attrition_db_customer_summary.xlsx
│
├── /data_prep
│     ├── setup.sql
│     ├── cleaned_company_profiles.sql
│     ├── cleaned_subscriptions.sql
│     ├── cleaned_support_tickets.sql
│     ├── cleaned_usage_events.sql
│     └── eda.sql
│
├── /ml_deployment
│     ├── app.py
│     ├── MachineLearning.ipynb
│     ├── churn_rf_model.pkl
│     ├── model_columns.pkl
│     └── /.streamlit
│
└── /dashboard
      ├── B2B SaaS Churn Dashboard.pbix
      └── B2B SaaS Churn Dashboard Screenshot.png
```

---

## Sample SQL <a name="sql"></a>

<details>
<summary>Click to expand a sample query (champion reliance calculation)</summary>

```sql
-- INSIGHT: Champion Reliance Percentage
-- Measures how much of an account's total activity comes from its single most active user
-- Business impact: high reliance = single point of failure risk

WITH user_counts AS
         (SELECT s.company_id,
                 u.user_id,
                 s.churn_flag,
                 COUNT(u.event_id) AS user_events
          FROM staging_subscriptions s
                   JOIN staging_usage_events u
                        ON s.company_id = u.company_id
          GROUP BY s.company_id, s.churn_flag, u.user_id),
     company_totals AS (SELECT user_counts.company_id,
                               MAX(user_events)                          AS most_active_user,
                               SUM(user_events)                          AS total_company_events,
                               MAX(user_events) / SUM(user_events) * 100 AS champion_reliance_percentage,
                               churn_flag
                        FROM user_counts
                        GROUP BY user_counts.company_id, churn_flag)
SELECT churn_flag,
       ROUND(AVG(champion_reliance_percentage), 2) AS avg_champion_reliance
FROM company_totals
GROUP BY churn_flag;
-- INSIGHT 10: The Single Point of Failure
-- Observation: In retained companies, the most active user accounts
-- for about 22% of total usage. In churned companies, that jumps
-- to 33%.
--
-- Business Impact: Churned accounts lean too heavily on one
-- champion. If that person loses interest or leaves, there's no
-- one else driving adoption — and the account cancels.
--
-- More champion reliance in companies likely to churn.
```

</details>

---

## Limitations & Future Work

- Dataset is synthetic (~350 companies). Patterns are built to be realistic, but the sample size means model metrics should be read as a general direction, not a production-grade guarantee.
- Future versions could test the annual-billing incentive as a controlled experiment instead of just a correlation.
- Real-time data refresh, instead of a static snapshot, would be a natural next step for a production version.

---

## Connect

Built by Aryan Nair. [LinkedIn](https://www.linkedin.com/in/aryannair767/) · [Portfolio](https://aryannair767.github.io/)
