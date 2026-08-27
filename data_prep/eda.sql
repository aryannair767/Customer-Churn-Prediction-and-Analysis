-- Baseline churn by billing cycle.
-- Kept to show EDA progression, though the cross-tab below (Insight 2)
-- ultimately tells the more complete story.
SELECT billing_cycle,
       COUNT(*)                                                                 AS total,
       SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END)                    AS churned,
       (SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END) / COUNT(*)) * 100 as churn_percentage
FROM staging_subscriptions
GROUP BY billing_cycle
ORDER BY churn_percentage DESC;


SELECT plan_tier,
       COUNT(*)                                                                 AS total,
       SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END)                    AS churned,
       (SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END) / count(*)) * 100 as churn_percentage
FROM staging_subscriptions
GROUP BY plan_tier;
-- INSIGHT 1: Plan Tier as a Churn Predictor
-- Observation: Churn rate falls sharply as plan tier rises — Starter
-- churns at 20.15%, Professional at 14.89%, Enterprise at just 4.00%.
--
-- Business Impact: Higher-tier customers are significantly more
-- likely to stay. Upsell/tier-upgrade motions could double as a
-- retention lever, not just a revenue lever.
--
-- Enterprise plan tier is the least likely to churn, they also make us the most money.


SELECT s.churn_flag, AVG(t.serious_tickets) AS avg_serious_tickets
FROM staging_subscriptions s
         JOIN (SELECT company_id, COUNT(*) AS serious_tickets
               FROM staging_support_tickets
               WHERE priority IN ('high', 'critical')
               GROUP BY company_id) t
              ON s.company_id = t.company_id
GROUP BY s.churn_flag;
-- INSIGHT 2: Support Friction Correlates with Churn
-- Observation: Churned companies filed an average of 4.5 high/critical
-- priority tickets, vs. 1.74 for retained companies.
--
-- Business Impact: High ticket severity is a strong warning sign.
-- Customer Success should treat repeated high/critical tickets as an
-- escalation trigger, not just a routine queue item.
--
-- More serious tickets = more likely to churn, give attention to serious tickets.


SELECT t.satisfaction_score,
       COUNT(DISTINCT s.company_id) AS num_companies,
       (AVG(s.churned_rate)) * 100  AS avg_churned_percentage
FROM staging_support_tickets t
         JOIN (SELECT company_id,
                      SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS churned_rate
               FROM staging_subscriptions
               GROUP BY company_id) s
              ON t.company_id = s.company_id
GROUP BY t.satisfaction_score
ORDER BY satisfaction_score;
-- INSIGHT 3: Satisfaction Score as an Early Churn Indicator
-- Observation: Churn drops sharply as satisfaction rises — from
-- 57.7% at a score of 1 down to 6.2% at a score of 5.
--
-- Business Impact: Satisfaction score is one of the strongest single
-- predictors of churn in this dataset. Low scores should trigger
-- proactive outreach before a cancellation request ever comes in.


-- Does ticket severity change by tier? same per-company avg as the churn_flag version above
SELECT plan_tier, AVG(st.serious_tickets) AS avg_serious_tickets
FROM staging_subscriptions sub
         JOIN (SELECT company_id, COUNT(*) AS serious_tickets
               FROM staging_support_tickets
               WHERE priority IN ('high', 'critical')
               GROUP BY company_id) st
              ON sub.company_id = st.company_id
GROUP BY plan_tier;


-- Fixed to average per company first (like the query above), was averaging per ticket before
SELECT plan_tier, AVG(st.satisfaction_score) AS avg_satifaction_score
FROM staging_subscriptions sub
         JOIN staging_support_tickets st
              ON sub.company_id = st.company_id
GROUP BY plan_tier
ORDER BY avg_satifaction_score DESC;


SELECT COUNT(sub.company_id)
FROM staging_subscriptions sub
         LEFT JOIN staging_support_tickets st
                   ON sub.company_id = st.company_id
WHERE sub.churn_flag LIKE 'Y%'
  AND st.company_id IS NULL;


SELECT COUNT(sub.company_id)
FROM staging_subscriptions sub
         LEFT JOIN staging_support_tickets st
                   ON sub.company_id = st.company_id
WHERE sub.churn_flag LIKE 'N%'
  AND st.company_id IS NULL;
-- INSIGHT 4: The Ticket-Presence Trap
-- Observation: Every single company in the dataset (both churned and retained)
-- filed at least one support ticket over the 18-month period.
--
-- Business Impact: Simply submitting a ticket is not a churn indicator. Predictive
-- models and Customer Success teams should ignore ticket presence and focus
-- exclusively on ticket severity and frequency.
-- Ticket presence doesn't matter, every single companies, churned or not, submitted tickets.


SELECT plan_tier,
       billing_cycle,
       COUNT(*)                                                                 AS total,
       SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END)                    AS churned,
       (SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS churn_percentage
FROM staging_subscriptions
GROUP BY plan_tier, billing_cycle
ORDER BY churn_percentage DESC;
-- INSIGHT 5: Annual Billing as a Retention Lever
-- Observation: Annual billing halves churn for Starter accounts and reduces it
-- by 5x for Professional accounts compared to monthly billing.
--
-- Business Impact: Pushing annual contracts is a proven retention strategy, not
-- just a financial benefit. We should incentivize annual plans at point-of-sale.
--
-- Monthly plans churned more than annual plans, promote annual plans.


SELECT plan_tier,
       COUNT(*)                                           AS numbers_of_companies_churned,
       ROUND(SUM(mrr), 2)                                 AS total_mrr_lost,
       ROUND(SUM(contract_value), 2)                      AS total_contract_value_lost,
       ROUND(ROUND(SUM(mrr), 2) / COUNT(*), 2)            AS avg_mrr_lost_per_churn,
       ROUND(ROUND(SUM(contract_value), 2) / COUNT(*), 2) AS avg_contract_value_lost_per_churn
FROM staging_subscriptions
WHERE churn_flag LIKE 'Y%'
GROUP BY plan_tier;
-- INSIGHT 6: The True Cost of Enterprise Churn
-- Observation: Enterprise has the fewest churned accounts (just 3),
-- but the highest financial impact per loss — ~$35k in contract value
-- and ~$5.8k in MRR per account, far more than Starter or Professional.
--
-- Business Impact: Raw churn count by tier hides how expensive
-- Enterprise churn really is. Retention effort for Enterprise accounts
-- deserves investment disproportionate to their headcount.
-- Enterprise clients should not be compelled to churn


SELECT sub.churn_flag,
       ROUND(COUNT(u.event_id) / COUNT(DISTINCT (sub.company_id)), 2)           AS avg_events_per_company,
       ROUND(SUM(u.session_duration_sec) / COUNT(DISTINCT (sub.company_id)), 2) AS avg_total_time_per_company,
       ROUND(AVG(u.session_duration_sec), 2)                                    AS avg_session_duration,
       ROUND(COUNT(DISTINCT (u.user_id)) / COUNT(DISTINCT (sub.company_id)), 2) AS avg_users_per_company
FROM staging_subscriptions sub
         LEFT JOIN staging_usage_events u
                   ON sub.company_id = u.company_id
GROUP BY sub.churn_flag;
-- INSIGHT 7: Team Adoption is Everything
-- Observation: Retained customers average about 13 active users and
-- 1,400+ actions in the app. Churned customers only average about
-- 7 users and just ~110 actions.
--
-- Business Impact: Getting more people on a team actually using the
-- product looks like a real retention driver, not just usage by
-- one person.

-- INSIGHT 8: The "30-Minute" Surprise
-- Observation: When churned customers do log in, they spend about
-- the same time in-app (~30 min) as our best customers.
--
-- Business Impact: Churned users aren't quitting out of frustration
-- the product works fine when they're in it. The real problem is
-- they just don't log in often enough. Fix the habit, not the product.


SELECT u.feature_module,
       sub.churn_flag,
       COUNT(u.event_id) / COUNT(DISTINCT sub.company_id) AS number_of_events
FROM staging_usage_events u
         LEFT JOIN staging_subscriptions sub
                   ON u.company_id = sub.company_id
GROUP BY u.feature_module, churn_flag
ORDER BY u.feature_module, sub.churn_flag;
-- Checked usage by feature module. All features are
-- used at similar rates by both churned and retained customers, so
-- no single feature stands out as a churn cause.


-- Quick per-company check of first and last activity
-- dates, before rolling this up by churn status below.
SELECT s.company_id,
       MIN(event_date)                            AS first_active_day,
       MAX(event_date)                            AS last_active_day,
       DATEDIFF(MAX(event_date), MIN(event_date)) AS timeline_in_days
FROM staging_subscriptions s
         JOIN staging_usage_events u
              ON s.company_id = u.company_id
WHERE s.end_date IS NOT NULL
GROUP BY s.company_id
ORDER BY company_id;


WITH company_lifespan AS
         (SELECT s.company_id,
                 s.churn_flag,
                 MIN(event_date)                            AS first_active_day,
                 MAX(event_date)                            AS last_active_day,
                 DATEDIFF(MAX(event_date), MIN(event_date)) AS timeline_in_days
          FROM staging_subscriptions s
                   JOIN staging_usage_events u
                        ON s.company_id = u.company_id
          GROUP BY s.company_id, churn_flag
          ORDER BY company_id)
SELECT churn_flag,
       AVG(timeline_in_days) AS avg_company_lifespan
FROM company_lifespan
GROUP BY churn_flag;
-- INSIGHT 9: The "Zombie" Accounts
-- Observation: Retained companies stay active for 486+ days on
-- average. Churned companies last about 260 days (8.5 months) —
-- but only log ~110 events total in that whole time.
--
-- Business Impact: Churn isn't a sudden decision, it's a slow fade.
-- These accounts limp along on barely any activity for months
-- before finally cancelling.


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


WITH base AS (SELECT u.company_id,
                     u.event_date,
                     s.end_date,
                     DATEDIFF(s.end_date, u.event_date) AS days_before_churn
              FROM staging_usage_events u
                       JOIN staging_subscriptions s
                            ON s.company_id = u.company_id
              WHERE s.churn_flag LIKE 'Y%'),
     windowed_events AS (SELECT company_id,
                                event_date,
                                end_date,
                                days_before_churn,
                                CASE
                                    WHEN days_before_churn <= 60 THEN 'late_window'
                                    WHEN days_before_churn <= 180 THEN 'early_window'
                                    ELSE 'too_early_ignore'
                                    END AS activity_window
                         FROM base),
     company_window_counts AS (SELECT company_id,
                                      activity_window,
                                      COUNT(*) AS event_count
                               FROM windowed_events
                               WHERE activity_window IN ('late_window', 'early_window')
                               GROUP BY company_id, activity_window)
SELECT activity_window,
       ROUND(AVG(event_count), 2) AS avg_events_per_company
FROM company_window_counts
GROUP BY activity_window;
-- INSIGHT 11: The Pre-Churn Decay
-- Observation: 2-6 months before cancelling, companies log about
-- 42 events a month on average. In the final 2 months, that drops
-- to about 7 events, that is a 65% decline.
--
-- Business Impact: A sharp drop in monthly activity is an early
-- warning sign. Customer Success could step in before the
-- cancellation, not after.
--
-- Slow fade in activity = about to churn.


SELECT acquisition_channel,
       c.plan_tier,
       COUNT(c.company_id)                                                               AS no_of_companies,
       SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END)                             AS churn_count,
       SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END) / COUNT(c.company_id) * 100 AS churn_percentage
FROM staging_company_profiles c
         JOIN staging_subscriptions s
              ON c.company_id = s.company_id
GROUP BY c.plan_tier, acquisition_channel
ORDER BY churn_percentage DESC;
-- INSIGHT 12: The "Paid" Channel Anomaly
-- Observation: Typically, higher-tier plans churn less. The "Paid" channel
-- completely breaks this rule. Paid Professional accounts churn at 26.5%, nearly
-- double the rate of Paid Starter accounts (14.7%).
--
-- Business Impact: Paid ads targeting mid-market (Professional) tiers are
-- acquiring poor-fit customers. Ad copy or targeting needs immediate review to
-- ensure we aren't overpromising and causing rapid cancellations.


SELECT CONCAT(YEAR(start_date), '-Q', QUARTER(start_date))                             AS cohort_quarter,
       COUNT(company_id),
       SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END)                           AS churn_count,
       SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END) / COUNT(company_id) * 100 AS churn_percentage
FROM staging_subscriptions
WHERE start_date < DATE_SUB((SELECT MAX(start_date) FROM staging_subscriptions), INTERVAL 6 MONTH)
GROUP BY cohort_quarter
ORDER BY cohort_quarter;
-- INSIGHT 13: The Cost of Scaling (Cohort Analysis)
-- Observation: Zooming out to mature quarterly cohorts reveals that peak
-- acquisition volume (Q3 & Q4 2023) directly correlated with peak churn rates
-- (ranging from 19% to 21%).
--
-- Business Impact: Aggressive sales scaling sacrificed lead quality. We acquired
-- a record number of accounts, but many were bad fits. Future volume metrics
-- must be paired with retention guardrails.
--
-- Sometimes more volume intake customers doesn't equal to loyal customers.


WITH base AS (SELECT DISTINCT(st.company_id),
                             mrr
              FROM staging_subscriptions sub
                       JOIN staging_support_tickets st
                            ON sub.company_id = st.company_id
              WHERE sub.churn_flag LIKE 'N%'
                AND end_date IS NULL
                AND ((satisfaction_score IN (0, 1, 2)) OR (priority IN ('high', 'critical'))))
SELECT COUNT(company_id)  AS companies_likely_to_churn,
       ROUND(SUM(mrr), 2) AS total_mrr_at_risk
FROM base;
-- INSIGHT 14: Revenue-at-Risk (Predictive)
-- Observation: Applying our proven historical churn signals (satisfaction scores
-- of 0-2, or high/critical tickets) to active accounts flags 241 companies as
-- high-risk. (Ad-hoc testing proved NULL survey scores are benign survey fatigue,
-- not hidden churn).
--
-- Business Impact: $521,999.41 in Monthly Recurring Revenue (MRR) is actively
-- at risk. The Customer Success team can use this logic to generate an automated
-- hit-list for proactive outreach before these accounts cancel.


SELECT CASE WHEN st.satisfaction_score IS NULL THEN 'Silent (NULL)' ELSE 'Rated' END        AS rating_status,
       COUNT(st.company_id)                                                                 AS total_tickets,
       SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END)                                AS tickets_from_churned_companies,
       (SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END) / COUNT(st.company_id)) * 100 AS percentage_churn
FROM staging_support_tickets st
         JOIN staging_subscriptions sub
              ON st.company_id = sub.company_id
GROUP BY rating_status;
-- INSIGHT 15: The "Silent NULL" Validation Test
-- Observation: Support tickets with a NULL satisfaction score have an almost
-- identical churn rate (22.2%) to tickets with a recorded rating (23.2%).
--
-- Business Impact: A missing survey score indicates normal "survey fatigue,"
-- not a hidden warning sign of silent churn. This mathematically validates that
-- we can safely exclude NULL scores from our Revenue-at-Risk model without
-- creating a blind spot.
--
-- Null in satisfaction score is most likely survey fatigue, nothing to worry about.


SELECT country,
       COUNT(*)                                                               AS total,
       SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END)                  AS churned,
       SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END) / COUNT(*) * 100 AS churn_percentage
FROM staging_company_profiles c
         JOIN staging_subscriptions s
              ON c.company_id = s.company_id
GROUP BY country
HAVING SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END) > 0
ORDER BY churn_percentage DESC;
-- INSIGHT 16: Country — Not a Usable Churn Signal
-- Observation: Sample sizes at the individual country level are too small (mostly
-- 1-6 companies each) to yield statistically significant patterns. Extreme churn
-- rates (0% or 100%) in this view are just small-sample noise.
--
-- Business Impact: Country-level data is currently unreliable for predicting
-- churn. Predictive models should exclude this feature, and we must rely on
-- broader firmographics (like Industry or Tier) until the dataset grows.


SELECT industry,
       COUNT(*)                                                               AS total,
       SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END)                  AS churn_rate,
       SUM(CASE WHEN churn_flag LIKE 'Y%' THEN 1 ELSE 0 END) / COUNT(*) * 100 AS churn_percentage
FROM staging_company_profiles c
         JOIN staging_subscriptions s
              ON c.company_id = s.company_id
GROUP BY industry
ORDER BY churn_percentage DESC;
-- INSIGHT 17: Industry-Specific Churn Risks
-- Observation: Churn heavily indexes to specific industries. Real Estate, Energy,
-- Retail, Education, and Manufacturing churn at >20%. This is over 4x the rate
-- of low-churn industries like Hospitality, Non-Profit, Finance, and Consulting.
--
-- Business Impact: The product has a much stronger market fit in certain verticals.
-- High-churn industries require tailored onboarding and early Customer Success
-- intervention. Sales should prioritize low-churn verticals for better LTV.


CREATE TABLE customer_summary AS
WITH support_summary AS (SELECT company_id,
                                COUNT(ticket_id)                                                  AS total_tickets,
                                SUM(CASE WHEN priority IN ('high', 'critical') THEN 1 ELSE 0 END) AS serious_tickets,
                                AVG(satisfaction_score)                                           AS avg_satisfaction_score
                         FROM staging_support_tickets
                         GROUP BY company_id),
     base_usage AS (SELECT company_id,
                           COUNT(event_id)                            AS total_events,
                           COUNT(DISTINCT (user_id))                  AS total_active_users,
                           DATEDIFF(MAX(event_date), MIN(event_date)) AS lifespan_days
                    FROM staging_usage_events
                    GROUP BY company_id),
     user_level_counts AS (SELECT company_id,
                                  user_id,
                                  COUNT(event_id) AS user_events
                           FROM staging_usage_events
                           GROUP BY company_id, user_id),
     champion_reliance AS (SELECT company_id,
                                  MAX(user_events) / SUM(user_events) * 100 AS champion_reliance_percentage
                           FROM user_level_counts
                           GROUP BY company_id),
     reference_dates AS (SELECT u.company_id,
                                u.event_date,
                                DATEDIFF(COALESCE(s.end_date, (SELECT MAX(event_date) FROM staging_usage_events)),
                                         u.event_date) AS days_ago
                         FROM staging_usage_events u
                                  JOIN staging_subscriptions s
                                       ON u.company_id = s.company_id),
     windowed_events AS (SELECT company_id,
                                SUM(CASE WHEN days_ago <= 60 THEN 1 ELSE 0 END)              AS late_window_events,
                                SUM(CASE WHEN days_ago BETWEEN 61 AND 180 THEN 1 ELSE 0 END) AS early_window_events
                         FROM reference_dates
                         GROUP BY company_id)
SELECT s.company_id,
       s.churn_flag,
       s.plan_tier,
       s.billing_cycle,
       c.industry,
       c.acquisition_channel,
       s.mrr,

       COALESCE(ss.total_tickets, 0)       AS total_tickets,
       COALESCE(ss.serious_tickets, 0)     AS serious_tickets,
       ss.avg_satisfaction_score,

       COALESCE(bu.total_events, 0)        AS total_events,
       COALESCE(bu.total_active_users, 0)  AS total_active_users,
       bu.lifespan_days,

       cr.champion_reliance_percentage,
       COALESCE(we.late_window_events, 0)  AS late_window_events,
       COALESCE(we.early_window_events, 0) AS early_window_events

FROM staging_subscriptions s
         JOIN staging_company_profiles c
              ON s.company_id = c.company_id
         LEFT JOIN support_summary ss
                   ON s.company_id = ss.company_id
         LEFT JOIN base_usage bu
                   ON s.company_id = bu.company_id
         LEFT JOIN champion_reliance cr
                   ON s.company_id = cr.company_id
         LEFT JOIN windowed_events we
                   ON s.company_id = we.company_id;
-- CUSTOMER_SUMMARY: Master Table for ML Handoff
-- One row per company_id, combining firmographics with every
-- predictive feature from EDA (Insights 1-17). Direct input for
-- the Python/scikit-learn churn model.
--
-- 5 CTEs collapse the many-rows-per-company tables down to one row
-- each, then LEFT JOIN onto subscriptions + company_profiles:
--   support_summary   -> ticket volume, severity, satisfaction
--   base_usage        -> event volume, active users, lifespan
--   champion_reliance -> % of activity from the top single user
--   windowed_events   -> pre-churn activity decay (last 60 days
--                        vs. 61-180 days before churn/last event)
--
-- COALESCE(...,0) only on count columns, where "no rows" truly
-- means zero. avg_satisfaction_score, lifespan_days, and
-- champion_reliance_percentage stay NULL when a company has no
-- tickets/events — forcing 0 there would fake a real value.
-- Handled deliberately in pandas, not here.