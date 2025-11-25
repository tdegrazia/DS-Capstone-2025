import plotly.express as px
import pandas as pd


def risk_card_html(risk: float) -> str:
    """Simple neutral risk card (no red/green)."""
    return f"""
    <div style="padding:16px;border-radius:14px;
                background:rgba(255,255,255,0.07);
                border:1px solid rgba(255,255,255,0.15);">
      <h3 style="margin:0 0 8px 0;font-size:18px;">Estimated Risk (0 – 1)</h3>
      <div style="font-size:40px;font-weight:700;">{risk:.2f}</div>
      <p style="font-size:12px;opacity:0.8;margin-top:6px;">
        Prototype estimate; interpret alongside clinical context and uncertainty.
      </p>
    </div>
    """


def simple_trend(df: pd.DataFrame, xcol: str, ycol: str, title: str):
    if xcol not in df.columns or ycol not in df.columns:
        return None
    fig = px.line(df, x=xcol, y=ycol, title=title, template="plotly_white")
    return fig

def wearable_summary(patient: pd.Series) -> pd.DataFrame:
    """Return a Metric/Value table for one wearable patient row."""
    summary_data = {
        "Age": patient.get("age", "N/A"),
        "Gender": patient.get("gender", "N/A"),
        "Height (cm)": patient.get("height", "N/A"),
        "Weight (kg)": patient.get("weight", "N/A"),
        "Avg Heart Rate (bpm)": patient.get("hear_rate", "N/A"),
        "Resting Heart Rate": patient.get("resting_heart", "N/A"),
        "Steps per Day": patient.get("steps", "N/A"),
        "Calories Burned": patient.get("calories", "N/A"),
        "Distance (m)": patient.get("distance", "N/A"),
        "Activity Level": patient.get("activity", "N/A"),
    }
    return pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])


def wearable_insights(patient: pd.Series) -> list[str]:
    """Return a list of bullet-point lifestyle insights for one row."""
    insights: list[str] = []

    if patient.get("resting_heart", 0) > 80:
        insights.append("• Elevated resting heart rate — may indicate stress or low cardiovascular fitness.")
    if patient.get("steps", 0) < 5000:
        insights.append("• Low daily activity detected; may benefit from increased movement.")
    if patient.get("steps", 0) > 10000:
        insights.append("• Excellent daily activity — strong daily activity level.")
    if patient.get("hear_rate", 0) > 110:
        insights.append("• High average heart rate; monitor for overtraining or stress.")

    if not insights:
        insights.append("• Heart rate and activity within healthy ranges.")

    return insights

