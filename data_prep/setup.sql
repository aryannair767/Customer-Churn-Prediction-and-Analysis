-- I will create staging tables as backup in case I mess up.

CREATE TABLE customer_attrition_db.staging_company_profiles
AS SELECT * FROM customer_attrition_db.company_profiles;

CREATE TABLE customer_attrition_db.staging_subscriptions
AS SELECT * FROM customer_attrition_db.subscriptions;

CREATE TABLE customer_attrition_db.staging_support_tickets
AS SELECT * FROM customer_attrition_db.support_tickets;

CREATE TABLE customer_attrition_db.staging_usage_events
AS SELECT * FROM customer_attrition_db.usage_events;

-- Drop table and recreate them in-case of any errors.
DROP TABLE customer_attrition_db.staging_company_profiles;

DROP TABLE customer_attrition_db.staging_usage_events;

DROP TABLE customer_attrition_db.staging_subscriptions;