import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# --- Streamlit Page Settings ---
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("House Price Prediction App")
st.write("Enter the details of the house in the sidebar to get an estimated price.")

# --- Load Data and Train Model ---
@st.cache_resource
def load_data_and_train_model():

    # Load dataset (change path if needed)
    df = pd.read_csv(r"D:\TOPS_DS\house_price_regression_dataset.csv")

    # Features and Target
    X = df.drop('House_Price', axis=1)
    y = df['House_Price']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, mae, rmse, r2, X


model, mae, rmse, r2, X = load_data_and_train_model()

# --- Model Performance ---
st.subheader("Model Performance on Test Data")

st.info(f"Mean Absolute Error (MAE): ${mae:,.2f}")
st.info(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
st.info(f"R² Score: {r2:.2f}")

# --- Sidebar Inputs ---
st.sidebar.subheader("Enter House Features")

with st.sidebar.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        square_footage = st.number_input(
            "Square Footage", min_value=500, max_value=10000, value=2500, step=100
        )

        num_bedrooms = st.number_input(
            "Number of Bedrooms", min_value=1, max_value=10, value=3
        )

        num_bathrooms = st.number_input(
            "Number of Bathrooms", min_value=1, max_value=5, value=2
        )

        year_built = st.number_input(
            "Year Built", min_value=1800, max_value=2024, value=2005
        )

    with col2:
        lot_size = st.number_input(
            "Lot Size (acres)", min_value=0.1, max_value=10.0, value=0.75, step=0.05
        )

        garage_size = st.number_input(
            "Garage Size (cars)", min_value=0, max_value=5, value=2
        )

        neighborhood_quality = st.slider(
            "Neighborhood Quality (1-10)", min_value=1, max_value=10, value=7
        )

    predict_button = st.form_submit_button("Predict House Price")

# --- Prediction ---
if predict_button:

    new_house_data = {
        'Square_Footage': [square_footage],
        'Num_Bedrooms': [num_bedrooms],
        'Num_Bathrooms': [num_bathrooms],
        'Year_Built': [year_built],
        'Lot_Size': [lot_size],
        'Garage_Size': [garage_size],
        'Neighborhood_Quality': [neighborhood_quality]
    }

    new_house_df = pd.DataFrame(new_house_data)

    predicted_price = model.predict(new_house_df)[0]

    st.subheader("Predicted House Price")
    st.success(f"Estimated Price: **${predicted_price:,.2f}**")
