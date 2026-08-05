from xml.parsers.expat import model

import numpy as np
import pandas as pd
import joblib
import json

from tensorflow.keras.models import load_model

class Predictor:
    def __init__(self, model_dir="model"):
        self.model_dir = model_dir

        # Load model
        self.model = load_model(f"{model_dir}/inflation.keras")

        # Load scaler
        self.scaler_X = joblib.load(f"{model_dir}/scaler_X.pkl")
        self.scaler_y = joblib.load(f"{model_dir}/scaler_y.pkl")

        # Load metadata
        with open(f"{model_dir}/metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.time_step = self.metadata["time_step"]

        # load original dataset
        self.train_data = pd.read_csv(
            f"{model_dir}/training_dataset.csv"
        )

        (self.inflation_change_data, self.current_account_data, self.combined_data) = self._get_training_sequence()

    def _get_training_sequence(self):
        """
        recreate combined_data
        """

        inflation_change_data = self.train_data.iloc[13, 2:47].astype(float).values
        current_account_data = self.train_data.iloc[23, 2:47].astype(float).values

        combined_data = np.column_stack((inflation_change_data, current_account_data))

        return (inflation_change_data, current_account_data, combined_data)

    MAX_FORECAST_YEARS = 150

    def forecast_future(
        self, 
        n_years : int, 
        noise_std : float = 0.5
    ) -> list[dict] :

        # ตรวจสอบว่าค่า n_years
        if n_years < 1 or n_years > self.MAX_FORECAST_YEARS:
            raise ValueError(f"n_years ต้องอยู่ระหว่าง 1 ถึง {self.MAX_FORECAST_YEARS}")
        
        future_predictions = []

        # เริ่มจากข้อมูลล่าสุด
        last_data = self.combined_data[-self.time_step:]

        for i in range(n_years):
            # ปรับขนาดข้อมูลให้สอดคล้องกับ scaler
            last_data_scaled = self.scaler_X.transform(last_data)
            last_data_scaled = np.reshape(last_data_scaled, (1, self.time_step, last_data_scaled.shape[1]))
            
            # คาดการณ์ปีถัดไป
            next_year_scaled = self.model.predict(last_data_scaled, verbose=0)
            next_year = self.scaler_y.inverse_transform(next_year_scaled)[0][0]
            
            # เพิ่มความผันผวนแบบสุ่ม
            volatility = np.random.normal(0, noise_std)  # สร้างความผันผวนแบบสุ่ม
            next_year += volatility  # ผสมผลคาดการณ์กับความผันผวน

            # เก็บผลการคาดการณ์
            future_predictions.append(float(next_year))
            
            # ปรับ last_data เพื่อใช้ในการคาดการณ์ปีถัดไป
            next_current_account = self.current_account_data[len(self.inflation_change_data) + i] if len(self.inflation_change_data) + i < len(self.current_account_data) else float(next_year) * 10
            next_data = np.append(last_data[1:], [[float(next_year), float(next_current_account)]], axis=0)
            last_data = next_data

        return future_predictions


    def calculate_future_value(
        self, 
        initial_amount: float, 
        future_predictions: list[float]
    ) -> list[dict]:
        
        # คำนวณเงินหลังจากการบวกเงินเฟ้อในแต่ละปี
        amount = initial_amount
        result = []  # เพื่อเก็บจำนวนเงินของแต่ละปี
    
        for year, inflat_rate in enumerate(future_predictions, start=1):
            inflat_rate = inflat_rate / 100  # เปลี่ยนจาก % เป็นทศนิยม
            amount += amount*inflat_rate # บวกเปอร์เซ็นต์เงินเฟ้อในปีนั้น
            result.append({
                'year': year,
                'inflation_rate': inflat_rate,
                'amount': round(amount,2) # ปัดเศษให้เหลือ 2 ตำแหน่ง
            })
        
        return result