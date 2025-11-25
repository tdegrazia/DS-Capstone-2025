-- Users (for later, can be empty now)
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY,
  email TEXT UNIQUE,
  role TEXT CHECK (role IN ('patient', 'practitioner')) NOT NULL
);

-- Patients (one row per patient / session)
CREATE TABLE IF NOT EXISTS patients (
  id INTEGER PRIMARY KEY,
  name TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Files uploaded per patient
CREATE TABLE IF NOT EXISTS uploads (
  id INTEGER PRIMARY KEY,
  patient_id INTEGER NOT NULL,
  kind TEXT CHECK (kind IN ('genomics','wearables','ehr')) NOT NULL,
  filename TEXT,
  uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (patient_id) REFERENCES patients(id)
);

-- Model outputs per patient
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY,
  patient_id INTEGER NOT NULL,
  model_name TEXT NOT NULL,
  risk_score REAL,
  details_json TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (patient_id) REFERENCES patients(id)
);
