import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(
    model, 
    X_test, 
    y_test, 
    scaler_y,
):

    # คาดการณ์ข้อมูล
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_inv = scaler_y.inverse_transform(y_test)

    # วัด Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test_inv, y_pred)
    print(f'Mean Absolute Error (MAE): {mae:.2f}')

    # วัด Mean Squared Error (MSE)
    mse = mean_squared_error(y_test_inv, y_pred)
    print(f'Mean Squared Error (MSE): {mse:.2f}')

    # วัด Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')


def evaluate_accuracy(
    predicted_full,
    inflation_change_data,
    future_predictions
):
    # กำหนดเปอร์เซ็นต์ความแตกต่างที่ยอมรับได้ (เช่น 10%)
    tolerance = 0.03

    # คำนวณจำนวนที่คาดการณ์ "ถูกต้อง"
    correct_predictions = np.sum(np.abs(predicted_full - inflation_change_data) / np.abs(inflation_change_data) <= tolerance)

    # คำนวณ Accuracy เป็นเปอร์เซ็นต์
    accuracy = correct_predictions / len(inflation_change_data) * 100
    print(f'Accuracy: {accuracy:.2f}%')

    # คำนวณ MAE และ RMSE ระหว่างข้อมูลจริงและการคาดการณ์ historical data
    if len(predicted_full) == len(inflation_change_data):
        mae = mean_absolute_error(inflation_change_data, predicted_full)
        rmse = np.sqrt(mean_squared_error(inflation_change_data, predicted_full))
        print(f'Mean Absolute Error (MAE): {mae:.4f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    else:
        print(f"Shape mismatch: predicted_full ({len(predicted_full)}) vs inflation_change_data ({len(inflation_change_data)})")