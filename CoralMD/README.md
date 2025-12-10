# CoralMD: Local Setup and Run Instructions

CoralMD is a Streamlit-based prototype dashboard for multimodal personalized medicine. This folder contains all code and example data needed to run the app locally.

## Requirements

- Python 3.9+
- pip
- A virtual environment manager such as venv or conda

All Python dependencies are listed in `requirements.txt`.

## Quickstart: Running CoralMD Locally

1. Navigate into the CoralMD directory

    cd CoralMD

2. Create and activate a virtual environment (macOS / Linux)

    python3 -m venv .venv
    source .venv/bin/activate

   On Windows (PowerShell):

    python -m venv .venv
    .venv\Scripts\Activate

3. Install dependencies

    pip install -r requirements.txt

4. Launch the Streamlit app

    streamlit run app.py

5. Open the app in your browser

    http://localhost:8501

The Streamlit server will automatically reload when you edit the code.

## Project Structure

app.py               # Main Streamlit entry point
pages/               # Multi-page views (EHR, Genomics, Wearables, etc.)
data/                # Synthetic patient datasets
utils/               # Visualization and model utilities
coralmd.sqlite3      # Example SQLite database
requirements.txt     # Python dependencies

## Notes

- All data included in this directory is synthetic and intended only for demonstration.
- No external services or cloud infrastructure are required; the entire system runs locally.
- The application is designed as a prototype for educational and research purposes and is not a clinical tool.
