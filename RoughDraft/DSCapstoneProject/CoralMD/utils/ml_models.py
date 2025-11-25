import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def compute_mock_risk(df: pd.DataFrame):
    """
    Toy risk score between 0 and 1 based on available columns.
    Looks for hr_resting, glucose, cholesterol if present.
    """
    preferred = ["hr_resting", "glucose", "cholesterol"]
    cols = [c for c in preferred if c in df.columns]

    if len(cols) >= 2:
        X = df[cols].fillna(df[cols].median())
        y = np.linspace(0.1, 0.9, len(X))  # fake targets just to fit something
        model = RandomForestRegressor(max_depth=3, random_state=42)
        model.fit(X, y)
        last = X.iloc[-1].to_numpy().reshape(1, -1)
        risk = float(np.clip(model.predict(last)[0], 0, 1))
        details = {"features_used": cols, "last_row": X.iloc[-1].to_dict(), "heuristic": False}
    else:
        # fallback heuristic
        hr = float(df.get("hr_resting", pd.Series([60])).iloc[-1])
        glu = float(df.get("glucose", pd.Series([95])).iloc[-1])
        chol = float(df.get("cholesterol", pd.Series([180])).iloc[-1])
        risk = float(np.clip((hr / 100 + glu / 200 + chol / 300) / 3, 0, 1))
        details = {"heuristic": True, "hr": hr, "glucose": glu, "cholesterol": chol}

    return risk, details
