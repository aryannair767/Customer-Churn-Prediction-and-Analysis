-- Standardize column data types to match the columns contents.
ALTER TABLE staging_subscriptions
MODIFY COLUMN start_date DATE,
MODIFY COLUMN end_date DATE,
MODIFY COLUMN mrr DOUBLE,
MODIFY COLUMN contract_value DOUBLE;


UPDATE staging_subscriptions
SET plan_tier = CASE
    WHEN plan_tier LIKE 'Star%' THEN 'starter'
    WHEN plan_tier LIKE 'Pro%' THEN 'professional'
    WHEN plan_tier LIKE 'Enter%' THEN 'enterprise'
END;