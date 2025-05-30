# **AI-based Crop Recommendation and Price Prediction**

## **Description**
This project assists farmers by recommending the three most suitable crops for a given piece of land based on soil and environmental factors. Additionally, it predicts the future prices of these recommended crops using historical price data, helping farmers make informed decisions for better profitability.

## **Features**
- 🌱 **Crop Recommendation**: AI suggests the top three suitable crops based on soil parameters, climate, and historical yield data.  
- 📊 **Price Prediction**: Forecasts future prices of the recommended crops using machine learning models trained on historical price trends.  
- 📈 **Data-Driven Decision Making**: Helps farmers choose the best crop by considering both suitability and potential profitability.  

## **Tech Stack**
- **Machine Learning & AI:** Scikit-learn, TensorFlow/Keras, Pandas, NumPy
- **Programming Language:** Python
- **Backend:** Flask
- **Frontend:** HTML, CSS, and JavaScript
 

## **Dataset**
- **For crop recommendation :** Nitrogen, Phosporous, Potassium, Temperature, Humidity, pH, Rainfall, Crop Name
- **For price prediction :** Crop Name, Date, Min Price, Max Price, Modal Price

## **How It Works**
1. **User Input**: Farmers provide soil, climate, and location details.  
2. **Crop Recommendation Model**: AI suggests three suitable crops.  
3. **Price Prediction Model**: Forecasts future prices of the recommended crops.  
4. **Output**: The system displays the recommended crops along with their predicted price trends.  

- we have choosen two models for this project.
- 1st model is for the crop recommendation model we are planning for ensembling model(Stacking Classifier).
- 2nd model is for the crop price prediction we are planning for the time series model(LSTM).

-deployment link of our project : http://15.207.98.207:5000/
