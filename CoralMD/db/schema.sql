-- will run in Step 3
CREATE TABLE IF NOT EXISTS patients (
  id SERIAL PRIMARY KEY,
  name TEXT,
  age INT,
  gender TEXT,
  hr_resting FLOAT,
  glucose FLOAT,
  cholesterol FLOAT,
  risk_score FLOAT
);
