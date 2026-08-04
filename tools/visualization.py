import matplotlib.pyplot as plt
import numpy as np


def plot_prediction(    
    model,
    scaler_X,
    scaler_y,
    combined_data,
    time_step,
    inflation_change_data,
    current_account_data,
    n_years, 
    future_predictions
):

    # พล็อตกราฟการคาดการณ์ตามจำนวนปีที่กำหนด
    plt.figure(figsize=(10, 6))

    # พล็อตข้อมูลจริง (% การเปลี่ยนแปลงของอัตราเงินเฟ้อ)
    plt.scatter(np.arange(1, len(inflation_change_data) + 1), inflation_change_data, color='blue', label='Actual Inflation Change')

    # ตรวจสอบขนาดของ combined_data ก่อน reshape
    print(f"Original combined data shape: {combined_data.shape}")

    # ตรวจสอบว่าจำนวนข้อมูลเพียงพอสำหรับการสร้างลำดับตาม time step หรือไม่
    if len(combined_data) > time_step:
        # แปลงข้อมูลเต็มรูปแบบและปรับขนาดใหม่สำหรับ time step = 5
        X_full_scaled = scaler_X.transform(np.reshape(combined_data[:-1], (-1, combined_data.shape[1])))
        
        # แก้ไข reshape ให้ใช้เฉพาะข้อมูลที่สามารถแบ่งได้ตาม time step
        X_full_scaled = np.reshape(X_full_scaled[:len(X_full_scaled) - len(X_full_scaled) % time_step], (-1, time_step, combined_data.shape[1]))
        print(f"Reshaped X_full_scaled shape: {X_full_scaled.shape}")

        # ทำนายข้อมูลในอดีต (historical data) โดยใช้โมเดล
        predicted_full_scaled = model.predict(X_full_scaled)
        predicted_full = scaler_y.inverse_transform(predicted_full_scaled)

        # พล็อตกราฟสำหรับ historical data และการคาดการณ์
        plt.plot(np.arange(1, len(predicted_full) + 1), predicted_full, color='red', label='LSTM Prediction (Historical Data)')
    else:
        print(f"Not enough data for the chosen time_step ({time_step})")

    # พล็อตการคาดการณ์ในจำนวนปีที่กำหนด
    years = np.arange(len(inflation_change_data) + 1, len(inflation_change_data) + n_years + 1)
    plt.plot(years, future_predictions, color='orange', linestyle='--', label=f'LSTM Prediction (Next {n_years} Years)')

    # พล็อตจุดการคาดการณ์ในแต่ละปีถัดไป
    for i in range(n_years):
        plt.scatter(len(inflation_change_data) + i + 1, future_predictions[i], color='green', marker='x', s=100, label=f'Predicted Change for year {len(inflation_change_data) + i + 1}' if i == 0 else "")

   # เพิ่มชื่อและรายละเอียดกราฟ
    plt.title(f'% Inflation Rate Change Prediction using LSTM (Next {n_years} Years)')
    plt.xlabel('Year')
    plt.ylabel('% Inflation Change')
    plt.legend()
    plt.grid(True)
    plt.show() 

    return predicted_full

    