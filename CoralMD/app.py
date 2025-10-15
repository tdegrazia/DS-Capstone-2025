
**app.py** (a tiny “hello, world” Streamlit to prove the plumbing)
```python
import streamlit as st
import pandas as pd

st.set_page_config(page_title="CoralMD", layout="wide")
st.title("🌊 CoralMD — Precision Medicine Prototype")

st.write("If you can see this, your Streamlit setup works!")

# demo: show a tiny dataframe
demo = pd.DataFrame({"metric": ["HR_resting", "Glucose", "Cholesterol"],
                     "value": [58, 92, 178]})
st.dataframe(demo, use_container_width=True)
