import os
from flask import Flask,request,jsonify
import predictor

app = Flask(__name__)

predictor_class = predictor.Predictor()

@app.route('/predict', methods=['POST'])
def predict():

    # รับ input (จำนวนปีและจำนวนเงินเริ่มต้น) จาก request
    data = request.get_json()

    try:
        # แปลงค่า n_years และ initial_amount เป็น int
        n_years = int(data.get('n_years'))
        initial_amount = float(data.get('initial_amount'))
        forcast_rate = predictor_class.forecast_future(n_years=n_years, noise_std=0.5)
        predictions = predictor_class.calculate_future_value(initial_amount=initial_amount, future_predictions=forcast_rate)

    except ValueError as e:
        return jsonify({'Error': str(e)}), 400
    
    return jsonify(predictions), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(debug=False,host='0.0.0.0' ,port=port)



