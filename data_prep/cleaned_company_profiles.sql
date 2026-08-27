-- Standardize plan_tier column.
UPDATE customer_attrition_db.staging_company_profiles
SET plan_tier = CASE
    WHEN plan_tier like 's%' THEN 'starter'
    WHEN plan_tier like 'p%' THEN 'professional'
    WHEN plan_tier like 'e%' THEN 'enterprise'
    ELSE 'unknown'
END;

-- Standardize billing_cycle column.
UPDATE customer_attrition_db.staging_company_profiles
SET billing_cycle = CASE
    WHEN billing_cycle like 'a%' THEN 'annual'
    WHEN billing_cycle like 'm%' THEN 'monthly'
    ELSE 'unknown'
END;

-- Standardize customer_attrition column.
UPDATE customer_attrition_db.staging_company_profiles
SET acquisition_channel = CASE
        WHEN acquisition_channel like 'd%' THEN 'direct'
        WHEN acquisition_channel like 'part%' THEN 'partner'
        WHEN acquisition_channel like 'paid%' THEN 'paid'
        WHEN acquisition_channel like 'r%' THEN 'referral'
        WHEN acquisition_channel like 'o%' THEN 'organic'
        ELSE 'unknown'
    END;

-- Clean and standardize mixed dated formats.
UPDATE staging_company_profiles
SET signup_date = CASE
    WHEN signup_date LIKE '20__-%' THEN STR_TO_DATE(signup_date, '%Y-%m-%d')
    WHEN signup_date LIKE '__/__/___%' THEN STR_TO_DATE(signup_date, '%d/%m/%Y')
    WHEN signup_date LIKE '__-__-20%' THEN STR_TO_DATE(signup_date, '%m-%d-%Y')
    END;

ALTER TABLE staging_company_profiles
MODIFY COLUMN signup_date DATE;