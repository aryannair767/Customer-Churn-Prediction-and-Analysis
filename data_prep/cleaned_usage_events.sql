-- Convert event_date (text, DD/MM/YYYY) to real DATE
UPDATE staging_usage_events
SET event_date = CASE
WHEN event_date LIKE '__/__/20__%' THEN STR_TO_DATE(event_date, '%d/%m/%Y')
END;


-- Convert event_timestamp (text, DD/MM/YYYY HH:MM:SS) to real DATETIME
UPDATE staging_usage_events
SET event_timestamp = STR_TO_DATE(event_timestamp, '%d/%m/%Y %H:%i:%s');


-- Lock in correct column types
ALTER TABLE staging_usage_events
MODIFY COLUMN event_timestamp DATETIME,
MODIFY COLUMN session_duration_sec INT,
MODIFY COLUMN event_date DATE;

