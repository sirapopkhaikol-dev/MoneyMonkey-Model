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


    def forecast_future(self, n_years, noise_std=0.5):
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
            future_predictions.append(next_year)
            
            # ปรับ last_data เพื่อใช้ในการคาดการณ์ปีถัดไป
            next_current_account = self.current_account_data[len(self.inflation_change_data) + i] if len(self.inflation_change_data) + i < len(self.current_account_data) else next_year * 10
            next_data = np.append(last_data[1:], [[next_year, next_current_account]], axis=0)
            last_data = next_data

        return [
            {
                "year": i + 1,
                "inflation_change": prediction
            }
            for i, prediction in enumerate(future_predictions)
        ]


ref = Predictor()

results = ref.forecast_future(n_years=50)

print("Future Inflation Change Predictions for the Next 50 Years:", results)
# for result in results:
#     print(f"Year {result['year']}: {result['inflation_change']:.2f}%")