import numpy as np

def forecast_future(
    model,
    scaler_X,
    scaler_y,
    n_years,
    combined_data,
    time_step,
    inflation_change_data,
    current_account_data,
):

    future_predictions = []

    # ค่ามาตรฐานของ noise (สามารถปรับได้)
    noise_std = 0.5 # กำหนดค่าที่มากขึ้นหรือน้อยลงตามที่ต้องการ

    # เริ่มจากข้อมูลล่าสุด
    last_data = combined_data[-time_step:]

    for i in range(n_years):
        # ปรับขนาดข้อมูลให้สอดคล้องกับ scaler
        last_data_scaled = scaler_X.transform(last_data)
        last_data_scaled = np.reshape(last_data_scaled, (1, time_step, last_data_scaled.shape[1]))
        
        # คาดการณ์ปีถัดไป
        next_year_scaled = model.predict(last_data_scaled)
        next_year = scaler_y.inverse_transform(next_year_scaled)[0][0]
        
        # เพิ่มความผันผวนแบบสุ่ม
        volatility = np.random.normal(0, noise_std)  # สร้างความผันผวนแบบสุ่ม
        next_year += volatility  # ผสมผลคาดการณ์กับความผันผวน

        # เก็บผลการคาดการณ์
        future_predictions.append(next_year)
        
        # ปรับ last_data เพื่อใช้ในการคาดการณ์ปีถัดไป
        next_current_account = current_account_data[len(inflation_change_data) + i] if len(inflation_change_data) + i < len(current_account_data) else next_year * 10
        next_data = np.append(last_data[1:], [[next_year, next_current_account]], axis=0)
        last_data = next_data

        # แสดงผลการคาดการณ์
    for i, pred in enumerate(future_predictions):
        print(f'Predicted % Inflation Change for year {len(inflation_change_data) + i + 1}: {pred:.2f}')

    return n_years, future_predictions