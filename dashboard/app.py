"""Read-only research dashboard; run after generating model outputs."""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parents[1]
st.set_page_config(page_title="Crisis EWS Research", layout="wide")
st.title("Global Financial Crisis Early Warning System")
st.warning("This is an academic research model and not investment, economic-policy, or financial advice.")
st.caption("Historical, model-estimated risk only. It does not make claims about future crises or causality.")

model_files = list((ROOT / "results/model_outputs").glob("*_predictions.csv"))
if not model_files:
    st.info("No evaluated predictions yet. Run the documented pipeline first.")
    st.stop()
predictions = pd.concat([pd.read_csv(file) for file in model_files], ignore_index=True)
country = st.selectbox("Country", sorted(predictions["country_code"].unique()))
subset = predictions.loc[predictions["country_code"].eq(country)].sort_values("year")
st.line_chart(subset.pivot(index="year", columns="model", values="predicted_probability"))
st.dataframe(subset, use_container_width=True, hide_index=True)
