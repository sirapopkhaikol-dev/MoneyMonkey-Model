from src.dataset import load_data_from_google_sheet
from src.preprocessing import preprocess
# from src.forecast import forecast_future

# from tools.evaluator import evaluate, evaluate_accuracy
# from tools.visualization import plot_prediction
# from tools.calculator import calculate_future_value

from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

import os
import joblib
import json
from datetime import date
from pathlib import Path

EPOCHS = 100
BATCH_SIZE = 5
LEARNING_RATE = 0.001

df = load_data_from_google_sheet()

X_scaled, y_scaled, scaler_X, scaler_y, inflation_change_data, current_account_data, combined_data, time_step = preprocess(df)

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# ปรับรูปแบบข้อมูลให้เป็น [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# สร้างโมเดล LSTM ด้วยชั้น LSTM สองชั้น
model = Sequential()

# ชั้น LSTM แรก - ตั้งค่า return_sequences=True เพื่อส่งค่าผลลัพธ์ไปยังชั้นถัดไป
model.add(LSTM(units=200, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))

# ชั้น LSTM ที่สอง
model.add(LSTM(units=100))  # เปลี่ยนจำนวน units ได้ตามต้องการ

# ชั้น Dense สุดท้าย
model.add(Dense(1))

# ปรับ optimizer โดยกำหนดค่า learning rate ที่ต้องการ (เช่น 0.0005)
optimizer = Adam(learning_rate=LEARNING_RATE)

# คอมไพล์โมเดลด้วย optimizer ที่ปรับแล้ว
model.compile(optimizer=optimizer, loss='mean_squared_error')

# ฝึกโมเดล
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Save the model
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

model.save(MODEL_DIR / "inflation.keras")

joblib.dump(scaler_X, f"{MODEL_DIR}/scaler_X.pkl")
joblib.dump(scaler_y, f"{MODEL_DIR}/scaler_y.pkl")

metadata = {
    "dataset_version": date.today().isoformat(),
    "time_step": time_step,
    "feature_names": [
        "inflation_change",
        "current_account"
    ],
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "test_size":0.3,
    "random_state":42
}

with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

df.to_csv("model/training_dataset.csv", index=False)

print("Training completed.")
print("Artifacts saved to model/")

# -------------------------

# for test

# evaluater = evaluate(model, X_test, y_test, scaler_y)

# n_year__, future_predictions = forecast_future(model, scaler_X, scaler_y, n_years=50, combined_data=combined_data, time_step=time_step, inflation_change_data=inflation_change_data, current_account_data=current_account_data)

# predicted_full = plot_prediction(model, scaler_X, scaler_y, combined_data, time_step, inflation_change_data, current_account_data, n_year__, future_predictions)

# evaluater_accuracy = evaluate_accuracy(predicted_full, inflation_change_data, future_predictions)

# print([float(x) for x in future_predictions])

# finalpredict = [float(x) for x in future_predictions]

# result = calculate_future_value(n_year__, initial_amount=25000, finalpredict=finalpredict)

# print(result)

# -------------------------









