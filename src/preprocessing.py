import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess(df):
    # เลือกแถวที่ 14 และ 24 และแปลงค่าเป็น float
    inflation_change_data = df.iloc[13, 2:47].astype(float).values
    current_account_data = df.iloc[23, 2:47].astype(float).values

    # รวมข้อมูลทั้งสองปัจจัยเป็น 2D array
    combined_data = np.column_stack((inflation_change_data, current_account_data))

    # สร้างลำดับข้อมูล
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step)])
            y.append(data[i + time_step, 0])  # เป้าหมายคือ % การเปลี่ยนแปลงของอัตราเงินเฟ้อ
        return np.array(X), np.array(y)

    # ใช้ time_step = 1 เพื่อความง่าย
    time_step = 1
    X, y = create_dataset(combined_data, time_step)

    # ปรับขนาดข้อมูลให้อยู่ในช่วง [0, 1]
    scaler_X = MinMaxScaler(feature_range=(0, 10))
    scaler_y = MinMaxScaler(feature_range=(0, 10))
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    return X_scaled, y_scaled, scaler_X, scaler_y, inflation_change_data, current_account_data, combined_data, time_step