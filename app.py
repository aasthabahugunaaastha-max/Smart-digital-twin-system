import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px

st.set_page_config(layout="wide")

st.markdown("""
<style>
body { background-color: white; color: black; }
h1 { text-align:center; color:#00AEEF; }
h3 { color:#00AEEF; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>SMART DIGITAL TWIN SYSTEM</h1>", unsafe_allow_html=True)

df_full = pd.read_csv("data/real_time_data.csv")

if "index" not in st.session_state:
    st.session_state.index = 50

st.session_state.index += 1
if st.session_state.index > len(df_full):
    st.session_state.index = 50

df = df_full.iloc[:st.session_state.index]
df = df.tail(50)


temp = df["temperature"].iloc[-1]
battery = df["battery"].iloc[-1]
pressure = df["pressure"].iloc[-1]
anomaly = int(df["anomaly"].iloc[-1])

# ---------------- LINEAR MODEL ----------------
X = np.arange(len(df)).reshape(-1,1)
model = LinearRegression().fit(X, df["temperature"])
linear_line = model.predict(X)
pred_linear = model.predict([[len(df)]])[0]


try:
    from tensorflow.keras.models import load_model
    import joblib

    model_lstm = load_model("lstm_model.h5", compile=False)
    scaler = joblib.load("scaler.save")

    seq = df["temperature"].values[-10:].reshape(-1,1)
    seq = scaler.transform(seq)
    seq = seq.reshape(1,10,1)

    pred_lstm = scaler.inverse_transform(model_lstm.predict(seq))[0][0]

except:
    pred_lstm = pred_linear

mse_lr = mean_squared_error(df["temperature"], linear_line)
mse_lstm = (df["temperature"].iloc[-1] - pred_lstm) ** 2

status = "CRITICAL" if anomaly == 1 else "NORMAL"


risk = 0
if anomaly == 1:
    risk += 50

risk += min(max((temp-40)/40,0),1)*20
risk += min(max((pressure-1000)/50,0),1)*20
risk += min(max((60-battery)/60,0),1)*10
risk = round(min(risk,100),2)

rec = []

if anomaly == 1:
    rec.append("Anomaly detected")
if temp > 60:
    rec.append("Temperature too high")
if pressure > 1020:
    rec.append("Pressure abnormal")
if battery < 60:
    rec.append("Battery low")

if not rec:
    rec.append("System stable")

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("### Live Monitoring")

    fig, ax = plt.subplots(figsize=(10,4))
    x = np.arange(len(df))

    temp_n = (df["temperature"]-df["temperature"].min())/(df["temperature"].max()-df["temperature"].min())
    bat_n = (df["battery"]-df["battery"].min())/(df["battery"].max()-df["battery"].min())
    pres_n = (df["pressure"]-df["pressure"].min())/(df["pressure"].max()-df["pressure"].min())

    ax.plot(x, temp_n, label="Temp", linewidth=2)
    ax.plot(x, bat_n, label="Battery", linewidth=2)
    ax.plot(x, pres_n, label="Pressure", linewidth=2)

 
    for i in range(len(df)):
        if df["temperature"].iloc[i] > 60:
            ax.scatter(i, temp_n.iloc[i], s=70, color="red")
            ax.text(i+0.2, temp_n.iloc[i]+0.05, "HIGH", fontsize=8, color="red")

        if df["pressure"].iloc[i] > 1020:
            ax.scatter(i, pres_n.iloc[i], s=70, color="green")
            ax.text(i+0.2, pres_n.iloc[i]+0.05, "PRESS", fontsize=8, color="green")

        if df["battery"].iloc[i] < 60:
            ax.scatter(i, bat_n.iloc[i], s=70, color="orange")
            ax.text(i+0.2, bat_n.iloc[i]-0.08, "LOW", fontsize=8, color="orange")

    ax.legend()
    st.pyplot(fig)

    
    st.markdown("### Prediction Analysis")

    fig2, ax2 = plt.subplots(figsize=(10,4))

    ax2.plot(x, df["temperature"], label="Actual", linewidth=2)
    ax2.plot(x, linear_line, label="Linear", linestyle="--")


    ax2.scatter(len(df), pred_lstm, s=120, color="green", label="LSTM Future")
    ax2.text(len(df)+0.5, pred_lstm+0.5, "LSTM", fontsize=9, color="green")

    ax2.scatter(len(df), pred_linear, s=120, color="red", label="Linear Future")
    ax2.text(len(df)+0.5, pred_linear-0.5, "Linear", fontsize=9, color="red")

    ax2.set_title("Prediction Analysis")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Temperature")
    ax2.grid(alpha=0.3)

    ax2.legend()
    st.pyplot(fig2)

with col2:
    st.markdown("### System Overview")
    st.write(f"Temperature: {temp:.2f}")
    st.write(f"Battery: {battery:.2f}")
    st.write(f"Pressure: {pressure:.2f}")

    st.markdown("### Prediction")
    st.write(f"LSTM: {pred_lstm:.2f}")
    st.write(f"Linear: {pred_linear:.2f}")

    st.markdown("### Evaluation")
    st.write(f"LSTM MSE: {mse_lstm:.2f}")
    st.write(f"Linear Regression MSE: {mse_lr:.2f}")

    if mse_lstm < mse_lr:
        st.success("LSTM performs better on this data")
    else:
        st.warning("Linear model performs similarly or better")

    st.markdown("### Risk Score")
    st.progress(int(risk))

    if status == "CRITICAL":
        st.error(status)
    else:
        st.success(status)

    st.markdown("### Future Insight")

    if anomaly == 1 or risk > 70:
        st.error("System may become CRITICAL soon")
    elif pred_lstm > 60 or pred_linear > 60:
        st.warning("Rising trend detected")
    else:
        st.success("System will remain stable")

    st.markdown("### Recommendation")
    for r in rec:
        st.write("•", r)

st.markdown("### 3D Digital Twin")

fig3 = px.scatter_3d(
    df,
    x="temperature",
    y="battery",
    z="pressure",
    color="anomaly",
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("### AI Assistant")

query = st.text_input("Ask anything")

if query:
    st.write(f"System is {status} with risk {risk}%.\nTemp={temp}, Battery={battery}, Pressure={pressure}")

st.markdown("### Live Data")
st.dataframe(df.tail(10), use_container_width=True)

time.sleep(1)
st.rerun()

