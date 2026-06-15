import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Smart Water Quality Dashboard", layout="wide")

st.title("💧 Smart Water Quality Monitoring Dashboard")
st.write("Upload a CSV file containing water quality data.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Statistics")
    st.write(df.describe())

    # Sidebar filters
    st.sidebar.header("Dashboard Controls")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        selected_col = st.sidebar.selectbox("Select Parameter", numeric_cols)

        fig = px.line(df, y=selected_col, title=f"{selected_col} Trend")
        st.plotly_chart(fig, use_container_width=True)

    # Water Quality Index (simple formula)
    required = ["pH", "Temperature", "Turbidity"]

    if all(col in df.columns for col in required):

        df["WQI"] = (
            (100 - abs(df["pH"] - 7) * 10)
            + (100 - df["Temperature"])
            + (100 - df["Turbidity"])
        ) / 3

        st.subheader("Water Quality Index (WQI)")
        st.dataframe(df[["pH", "Temperature", "Turbidity", "WQI"]].head())

        avg_wqi = df["WQI"].mean()

        st.metric("Average WQI", round(avg_wqi, 2))

        if avg_wqi >= 80:
            st.success("✅ Water Quality: SAFE")
        elif avg_wqi >= 60:
            st.warning("⚠ Water Quality: MODERATE")
        else:
            st.error("🚨 Water Quality: UNSAFE")

        # Alerts
        st.subheader("Alerts")

        alerts = []

        if (df["pH"] < 6.5).any() or (df["pH"] > 8.5).any():
            alerts.append("Abnormal pH detected")

        if (df["Temperature"] > 35).any():
            alerts.append("High temperature detected")

        if (df["Turbidity"] > 50).any():
            alerts.append("High turbidity detected")

        if alerts:
            for alert in alerts:
                st.error(alert)
        else:
            st.success("No alerts detected")

        # Machine Learning Section
        st.subheader("Water Quality Prediction")

        df["Quality"] = np.where(df["WQI"] >= 80, 2,
                          np.where(df["WQI"] >= 60, 1, 0))

        X = df[["pH", "Temperature", "Turbidity"]]
        y = df["Quality"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)

        st.write(f"Model Accuracy: {acc:.2%}")

        st.subheader("Predict New Water Sample")

        ph = st.number_input("pH", value=7.0)
        temp = st.number_input("Temperature", value=25.0)
        turb = st.number_input("Turbidity", value=10.0)

        if st.button("Predict Quality"):
            pred = model.predict([[ph, temp, turb]])[0]

            if pred == 2:
                st.success("SAFE Water")
            elif pred == 1:
                st.warning("MODERATE Water")
            else:
                st.error("UNSAFE Water")

else:
    st.info("Upload a CSV file to begin.")
