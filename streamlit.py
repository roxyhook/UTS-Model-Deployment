import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# Load trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load encoders
with open('encoder_cat.pkl', 'rb') as file:
    encoder_cat = pickle.load(file)
with open('encoder_target.pkl', 'rb') as file:
    encoder_target = pickle.load(file)



def main():
    st.title("Hotel Booking Cancellation Prediction")

    # --- Numeric Inputs ---
    no_of_adults = st.number_input("Number of Adults", min_value=0, step=1)
    no_of_children = st.number_input("Number of Children", min_value=0, step=1)
    no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, step=1)
    no_of_week_nights = st.number_input("Number of Week Nights", min_value=0, step=1)
    required_car_parking_space = st.selectbox("Required Car Parking Space (0 = No, 1 = Yes)", [0, 1])
    lead_time = st.number_input("Lead Time", min_value=0, step=1)
    arrival_year = st.selectbox("Arrival Year", [2017, 2018])  
    arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
    arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))
    repeated_guest = st.selectbox("Repeated Guest (0 = No, 1 = Yes)", [0, 1])
    no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, step=1)
    no_of_previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, step=1)
    avg_price_per_room = st.number_input("Average Price per Room", min_value=0.0, step=1.0)
    no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, step=1)

    # --- Categorical Inputs ---
    type_of_meal_plan = st.selectbox("Type of Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    room_type_reserved = st.selectbox("Room Type Reserved", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
    market_segment_type = st.selectbox("Market Segment Type", ["Online", "Offline", "Corporate", "Complementary", "Aviation", "Other"])

    if st.button('Make Prediction'):
        features = [
            no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, 
            type_of_meal_plan, required_car_parking_space, room_type_reserved, lead_time, 
            arrival_year, arrival_month, arrival_date, market_segment_type, repeated_guest, 
            no_of_previous_cancellations, no_of_previous_bookings_not_canceled, avg_price_per_room, 
            no_of_special_requests
        ]
        result, prob_cancelled = make_prediction(features)

        if result == 1:
            # Not Cancelled
            st.markdown(f"""
            <div style="background-color:#d4edda; color:black; padding:10px; border-radius:5px; margin-bottom:10px;">
                <strong>✅ Prediction:</strong> Not Cancelled<br>
                <strong>📊 Probability of Cancellation:</strong> {prob_cancelled * 100:.2f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            # Cancelled
            st.markdown(f"""
            <div style="background-color:#f8d7da; color:black; padding:10px; border-radius:5px; margin-bottom:10px;">
                <strong>❌ Prediction:</strong> Cancelled<br>
                <strong>📊 Probability of Cancellation:</strong> {prob_cancelled * 100:.2f}%
            </div>
            """, unsafe_allow_html=True)

def make_prediction(features):
    categorical_cols = ['type_of_meal_plan','room_type_reserved', 'market_segment_type']
    
    # Prepare DataFrame for prediction
    input_data = pd.DataFrame([features], columns=[
        'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 
        'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 'lead_time', 
        'arrival_year', 'arrival_month', 'arrival_date', 'market_segment_type', 'repeated_guest', 
        'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 
        'no_of_special_requests'
    ])
    
    for col in categorical_cols:
        try:
            encoder = encoder_cat[col]
            input_data[col] = encoder.transform(input_data[col])
        except ValueError:
            input_data[col] = encoder.transform([encoder.classes_[0]])

    input_data = input_data.values.flatten().reshape(1, -1)
    
    prediction = model.predict(input_data)[0]
    prob_cancelled = model.predict_proba(input_data)[0][0]  

    return prediction, prob_cancelled

if __name__ == '__main__':
    main()
