-- Convert created_date & resolved_date (text, MM-DD-YYYY) to real dates
UPDATE staging_support_tickets
SET created_date = STR_TO_DATE(created_date, '%m-%d-%Y'),
resolved_date = STR_TO_DATE(resolved_date, '%m-%d-%Y');


-- Standardize priority values to lowercase full words
UPDATE staging_support_tickets
SET priority = CASE
           WHEN priority LIKE 'LO%' THEN 'low'
           WHEN priority LIKE 'MED%' THEN 'medium'
           WHEN priority LIKE 'HI%' THEN 'high'
           WHEN priority LIKE 'CR%' THEN 'critical'
    ELSE priority IS NULL
END;


-- Lock in correct column types
ALTER TABLE staging_support_tickets
MODIFY COLUMN created_date DATE,
MODIFY COLUMN resolved_date DATE,
MODIFY COLUMN satisfaction_score INT;