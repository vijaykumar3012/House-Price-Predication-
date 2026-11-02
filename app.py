"""Streamlit House Price Prediction App.  Run: streamlit run app.py"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_assets():
    best = joblib.load("models/best_model.pkl")
    lin = joblib.load("models/linear_model.pkl")
    rf = joblib.load("models/rf_model.pkl")
    with open("models/metadata.json") as f:
        meta = json.load(f)
    return best, lin, rf, meta


@st.cache_data
def load_data():
    return pd.read_csv("data/house_data.csv")


try:
    best_model, lin_model, rf_model, meta = load_assets()
except FileNotFoundError:
    st.error("Model not found. Run `python generate_data.py` then `python train.py` first.")
    st.stop()

data = load_data()


def fmt(lakh: float) -> str:
    """Format price given in Lakh as Lakh / Crore."""
    return f"₹{lakh / 100:.2f} Cr" if lakh >= 100 else f"₹{lakh:.2f} Lakh"


def make_row(area, bed, bath, loc, age, parking):
    return pd.DataFrame([{"Area": area, "Bedrooms": bed, "Bathrooms": bath,
                          "Location": loc, "Age": age,
                          "Parking": 1 if parking == "Yes" else 0}])


def rf_confidence(row):
    """Mean +/- 1 std of the individual trees = prediction confidence band."""
    pre, forest = rf_model.named_steps["pre"], rf_model.named_steps["model"]
    preds = np.array([t.predict(pre.transform(row)) for t in forest.estimators_])
    return preds.mean(), preds.std()


st.title("🏠 House Price Prediction App")
st.caption(f"Best model: **{meta['best_model']}** · "
           f"R² = {meta['results'][meta['best_model']]['R2']}")

tab_pred, tab_cmp, tab_data = st.tabs(["🔮 Predict", "⚖️ Compare Properties", "📊 Analytics"])

# ---------------- Predict ----------------
with tab_pred:
    c1, c2 = st.columns(2)
    with c1:
        area = st.number_input("Area (sq ft)", 300, 10000, 1200, step=50)
        bed = st.slider("Bedrooms", 1, 6, 3)
        bath = st.slider("Bathrooms", 1, 6, 2)
    with c2:
        loc = st.selectbox("Location", meta["locations"])
        age = st.slider("Age of Property (years)", 0, 50, 5)
        parking = st.radio("Parking Available", ["Yes", "No"], horizontal=True)

    if st.button("Predict Price", type="primary"):
        row = make_row(area, bed, bath, loc, age, parking)
        price = float(best_model.predict(row)[0])
        mean, std = rf_confidence(row)
        low, high = max(mean - 2 * std, 0), mean + 2 * std

        st.success(f"Predicted Price: **{fmt(price)}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Linear Regression", fmt(float(lin_model.predict(row)[0])))
        m2.metric("Random Forest", fmt(float(rf_model.predict(row)[0])))
        m3.metric("Likely Range (~95%)", f"{fmt(low)} – {fmt(high)}")

        report = row.assign(Predicted_Price_Lakh=round(price, 2),
                            Range_Low_Lakh=round(low, 2),
                            Range_High_Lakh=round(high, 2))
        st.download_button("⬇️ Download Report (CSV)",
                           report.to_csv(index=False).encode(),
                           "house_price_report.csv", "text/csv")

# ---------------- Compare ----------------
with tab_cmp:
    st.write("Enter two properties and compare predicted prices.")
    colA, colB = st.columns(2)
    inputs = []
    for col, label, d in [(colA, "Property A", (1200, 2, 2, 10, "Yes")),
                          (colB, "Property B", (1800, 3, 3, 5, "Yes"))]:
        with col:
            st.subheader(label)
            a = st.number_input("Area", 300, 10000, d[0], 50, key=label + "a")
            b = st.slider("Bedrooms", 1, 6, d[1], key=label + "b")
            ba = st.slider("Bathrooms", 1, 6, d[2], key=label + "ba")
            lo = st.selectbox("Location", meta["locations"], key=label + "l")
            ag = st.slider("Age", 0, 50, d[3], key=label + "ag")
            pk = st.radio("Parking", ["Yes", "No"], key=label + "p", horizontal=True)
            inputs.append((a, b, ba, lo, ag, pk))
    if st.button("Compare"):
        pa = float(best_model.predict(make_row(*inputs[0]))[0])
        pb = float(best_model.predict(make_row(*inputs[1]))[0])
        st.bar_chart(pd.DataFrame({"Price (Lakh)": [pa, pb]},
                                  index=["Property A", "Property B"]))
        st.info(f"A: **{fmt(pa)}**  |  B: **{fmt(pb)}**  |  Difference: **{fmt(abs(pa - pb))}**")

# ---------------- Analytics ----------------
with tab_data:
    st.subheader("Dataset Overview")
    st.dataframe(data.head(20), width="stretch")
    a, b = st.columns(2)
    with a:
        st.write("**Average Price by Location (Lakh)**")
        st.bar_chart(data.groupby("Location")["Price"].mean())
    with b:
        st.write("**Area vs Price**")
        st.scatter_chart(data, x="Area", y="Price", color="Location")
    st.write("**Model Performance**")
    st.table(pd.DataFrame(meta["results"]).T)
