import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = "lstm_model.keras"
SCALER_PATH = "scaler.pkl"
SEQ_LEN = 10   

def get_lstm_model(data):

  
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_model(MODEL_PATH, compile=False)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    # 🔹 Train new model
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1,1))

    X, y = [], []
    for i in range(SEQ_LEN, len(data_scaled)):
        X.append(data_scaled[i-SEQ_LEN:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(SEQ_LEN,1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, verbose=0)

  
    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler


def predict_lstm(model, scaler, data):

    # 🔹 If not enough data
    if len(data) < SEQ_LEN:
        return data[-1]

    seq = data[-SEQ_LEN:].reshape(-1,1)
    seq_scaled = scaler.transform(seq)

    pred = model.predict(seq_scaled.reshape(1,SEQ_LEN,1), verbose=0)

    return scaler.inverse_transform(pred)[0][0]
