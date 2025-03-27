from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

# Load datasets
price_data = pd.read_csv(r'C:\Users\India\Desktop\crop-recommendation-and-price-prediction\maize.csv')
df = pd.read_csv(r'C:\Users\India\Desktop\crop-recommendation-and-price-prediction\crop.csv')

# Preprocess data
price_data.columns = price_data.columns.str.strip()
price_data['Date'] = pd.to_datetime(price_data['Date'].str.strip(), dayfirst=True, errors='coerce')
price_data.dropna(subset=['Date'], inplace=True)
price_data = price_data.sort_values(by='Date')

crop_growth_time = {
    'Maize': 4, 'Wheat': 5, 'Rice': 6, 'Mango': 48,
    'Sugarcane': 21, 'Groundnut': 4, 'Green Chilly': 2,
    'Sunflower': 3, 'Turmeric': 8, 'Banana': 12,
    'Cotton': 7, 'Blackgram': 3, 'Tomato': 3,
}

# Encode and scale data
X = df.drop('label', axis=1)
y = df['label']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Feature Selection and Stacking Model
rfe_estimator = DecisionTreeClassifier(random_state=42)
from sklearn.feature_selection import RFE
rfe = RFE(estimator=rfe_estimator, n_features_to_select=7)
X_train_rfe = rfe.fit_transform(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()
dt = DecisionTreeClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)

estimators = [('knn', knn), ('nb', nb), ('dt', dt), ('svm', svm)]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=SVC(probability=True))
stacking_model.fit(X_train_rfe, y_train)

def recommend_top3_crops(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    input_rfe = rfe.transform(input_scaled)
    probabilities = stacking_model.predict_proba(input_rfe)
    top3_indices = probabilities[0].argsort()[-3:][::-1]
    top3_crops = label_encoder.inverse_transform(top3_indices)
    return [crop.capitalize() for crop in top3_crops]

def train_price_prediction_model(crop_name):
    crop_data = price_data[price_data['Crop'] == crop_name].copy()
    if crop_data.empty:
        return None

    crop_data.set_index('Date', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(crop_data[['Modal Price']])

    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    return model, scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        n = float(request.form['N'])
        p = float(request.form['P'])
        k = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        date_input = request.form['date']

        input_data = {
            'N': n, 'P': p, 'K': k,
            'temperature': temperature, 'humidity': humidity,
            'ph': ph, 'rainfall': rainfall
        }

        recommended_crops = recommend_top3_crops(input_data)

        # Maintain order by using a list instead of a dictionary
        predictions = []
        for crop in recommended_crops:
            result = train_price_prediction_model(crop)
            if result:
                model, scaler = result
                growth_time = crop_growth_time.get(crop, 0)
                target_date = pd.to_datetime(date_input) + pd.DateOffset(months=growth_time)
                recent_data = price_data[price_data['Crop'] == crop].tail(60)
                recent_data_scaled = scaler.transform(recent_data[['Modal Price']])
                X_test = recent_data_scaled.reshape(1, recent_data_scaled.shape[0], 1)

                predicted_scaled = model.predict(X_test)
                predicted_price = float(scaler.inverse_transform(predicted_scaled)[0][0])

                predictions.append({
                    'crop': crop,
                    'target_date': target_date.strftime('%Y-%m-%d'),
                    'predicted_price': predicted_price
                })

        return jsonify({'crops': recommended_crops, 'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
