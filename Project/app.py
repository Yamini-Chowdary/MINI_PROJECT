from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporous'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    # Add a dummy 8th feature to match the model's expected input
    extra_feature = 0  # Replace with actual value if known
    feature_list = [N, P, K, temp, humidity, ph, rainfall, extra_feature]
    
    single_pred = np.array(feature_list).reshape(1, -1)
    prediction = model.predict(single_pred)

    crop_dict = {
        1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut',
        6: 'Papaya', 7: 'Orange', 8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon',
        11: 'Grapes', 12: 'Mango', 13: 'Banana', 14: 'Pomegranate',
        15: 'Lentil', 16: 'Blackgrain', 17: 'Mugbean', 18: 'Mothbeans',
        19: 'Pigeonpeas', 20: 'Kidneybeans', 21: 'Chickpeas', 22: 'Coffee'
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated."
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)