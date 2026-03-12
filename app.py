# ==========================================
# GridX Pakistan – Streamlit App
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="GridX Pakistan", layout="wide")

st.title("🇵🇰 GridX Pakistan – AI Energy Market Intelligence")

# ==========================================
# LOAD DATA & MODEL
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("energy_data.csv", parse_dates=["datetime"], index_col="datetime")

@st.cache_resource
def load_model():
    return joblib.load("price_model.pkl")

df = load_data()
model = load_model()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================

st.sidebar.header("Simulation Controls")

battery_capacity = st.sidebar.slider("Battery Capacity (MWh)", 50, 500, 100)
power_limit = st.sidebar.slider("Power Limit (MW)", 5, 100, 20)

# ==========================================
# KPI SECTION
# ==========================================

col1, col2, col3 = st.columns(3)

col1.metric("Avg Load (MW)", round(df["load"].mean(),2))
col2.metric("Renewable Share (%)", round(((df["solar"]+df["wind"]).sum()/df["load"].sum())*100,2))
col3.metric("Avg Price", round(df["price"].mean(),2))

st.markdown("---")

# ==========================================
# MARKET DYNAMICS
# ==========================================

st.subheader("Load vs Renewables")
st.line_chart(df[["load", "solar", "wind"]].iloc[-168:])

st.subheader("Net Load")
st.line_chart(df["net_load"].iloc[-168:])

# ==========================================
# AI PRICE FORECAST
# ==========================================

st.subheader("AI Price Forecast (Next 24 Hours)")

future_hours = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=24, freq="H")

future_df = pd.DataFrame({
    "net_load": df["net_load"].iloc[-24:].values,
    "hour": future_hours.hour,
    "dayofweek": future_hours.dayofweek
})

forecast = model.predict(future_df)

forecast_series = pd.Series(forecast, index=future_hours)

combined = pd.concat([df["price"].iloc[-72:], forecast_series])

st.line_chart(combined)

# ==========================================
# BATTERY ARBITRAGE SIMULATION
# ==========================================

st.subheader("Battery Arbitrage Simulation")

soc = 0
profit = 0
soc_history = []
profit_history = []

mean_price = df["price"].mean()

for price in df["price"].iloc[-168:]:
    if price < mean_price*0.95:
        charge = min(power_limit, battery_capacity - soc)
        soc += charge
        profit -= charge*price
    elif price > mean_price*1.05:
        discharge = min(power_limit, soc)
        soc -= discharge
        profit += discharge*price
    soc_history.append(soc)
    profit_history.append(profit)

colA, colB = st.columns(2)
colA.metric("Final SOC", round(soc,2))
colB.metric("Total Profit", round(profit,2))

st.line_chart(pd.Series(soc_history))
st.line_chart(pd.Series(profit_history))

# ==========================================
# MARKET STATE DETECTOR
# ==========================================

st.subheader("Market Condition Detection")

latest_net_load = df["net_load"].iloc[-1]

if latest_net_load > df["net_load"].quantile(0.7):
    st.error("Scarcity Market — Buyer Active")
elif latest_net_load < df["net_load"].quantile(0.3):
    st.success("Surplus Market — Supplier Active")
else:
    st.info("Balanced Market")