# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict

# ---------------------------
# Page config & global style
# ---------------------------
st.set_page_config(
    page_title="UK Student Rental Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small CSS tweaks for cleaner look
st.markdown(
    """
    <style>
    .stApp { background-color: #f7fafc; }
    .big-title { font-size:30px; font-weight:700; }
    .section-title { font-size:18px; font-weight:600; color:#0f172a; }
    .card { background: white; padding: 1rem; border-radius: 12px; box-shadow: 0 4px 20px rgba(2,6,23,0.06); }
    .muted { color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Load assets (cached)
# ---------------------------
@st.cache_resource
def load_assets():
    """Load model, scaler and mapping files. Returns objects or raises an Exception."""
    # Try a couple of filenames gracefully
    tried = []
    try:
        # Model
        model = None
        model_name = "Unknown Model"
        possible_models = ['tuned_random_forest_model.joblib', 'tuned_lgbm_model.joblib']
        for path in possible_models:
            tried.append(path)
            if os.path.exists(path):
                model = joblib.load(path)
                model_name = os.path.splitext(os.path.basename(path))[0].replace('_', ' ').title()
                break
        if model is None:
            raise FileNotFoundError(f"No model file found. Tried: {tried}")

        # Preprocessing assets
        scaler = joblib.load('scaler.joblib') if os.path.exists('scaler.joblib') else None
        uni_distance_mapping = joblib.load('uni_distance_mapping.joblib') if os.path.exists('uni_distance_mapping.joblib') else {}
        station_distance_mapping = joblib.load('station_distance_mapping.joblib') if os.path.exists('station_distance_mapping.joblib') else {}
        feature_columns = joblib.load('feature_columns.joblib') if os.path.exists('feature_columns.joblib') else None

        if scaler is None or feature_columns is None:
            raise FileNotFoundError("Required preprocessing files (scaler or feature_columns) not found.")

        return model, scaler, uni_distance_mapping, station_distance_mapping, feature_columns, model_name
    except Exception as e:
        # Re-raise so the app can handle and show friendly message
        raise

try:
    model, scaler, uni_distance_mapping, station_distance_mapping, feature_columns, model_name = load_assets()
except Exception as e:
    st.sidebar.error(f"Error loading model/assets: {e}")
    st.stop()

# ---------------------------
# Constants and Options
# ---------------------------
CITY_OPTIONS = [
    'Nottingham', 'Liverpool', 'Coventry', 'Bristol', 'Birmingham', 'Manchester',
    'Sheffield', 'Leeds', 'Glasgow', 'Newcastle upon Tyne', 'Leicester',
    'Southampton', 'Edinburgh', 'London', 'Cardiff', 'Other_City'
]
PROPERTY_TYPES = ['flat', 'house', 'studio']

UNI_BINS = [-np.inf, 0.5, 2.0, 5.0, np.inf]
UNI_LABELS = ['Walking Distance', 'Short Commute', 'Medium Commute', 'Long Commute']
STATION_BINS = [-np.inf, 0.5, 1.5, 5.0, np.inf]
STATION_LABELS = ['Immediate Access', 'Short Walk', 'Medium Commute', 'Long Commute']
PLACEHOLDER_PRICE = 150.0  # for deposit ratio when real price unknown

# ---------------------------
# Feature engineering
# ---------------------------
def engineer_features(raw_input: Dict, feature_cols: list, uni_map: dict, station_map: dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Accept raw_input dictionary and return:
      - final_df: DataFrame with columns in model's expected order
      - analysis_flags: dict with engineered fields useful for UI display
    """
    data = raw_input.copy()
    # Engineered flags for UI
    analysis_flags = {}

    # Binary flags
    analysis_flags['Zero_Deposit_Required'] = 1 if data.get('deposit', 0) == 0 else 0
    data['Zero_Deposit_Required'] = analysis_flags['Zero_Deposit_Required']

    # Shared tenancy flag
    is_shared = data.get('rooms_available', 0) < data.get('Bedrooms', 1)
    is_multi_bed = data.get('Bedrooms', 1) > 1
    analysis_flags['Is_Shared_Tenancy'] = 1 if is_shared and is_multi_bed else 0
    data['Is_Shared_Tenancy'] = analysis_flags['Is_Shared_Tenancy']

    # Deposit to price ratio (placeholder)
    data['Deposit_to_Price_Ratio'] = data.get('deposit', 0) / (PLACEHOLDER_PRICE + 1e-6)

    # Distance categorical encoding
    uni_category = pd.cut([data.get('distance_to_uni_km', 0.0)], bins=UNI_BINS, labels=UNI_LABELS, right=True)[0]
    station_category = pd.cut([data.get('distance_to_station_km', 0.0)], bins=STATION_BINS, labels=STATION_LABELS, right=True)[0]
    data['uni_distance_encoded'] = uni_map.get(uni_category, 0)
    data['station_distance_encoded'] = station_map.get(station_category, 0)

    # One-hot for type (flat dropped)
    selected_type = data.pop('type', 'flat')
    data['type_house'] = 1 if selected_type == 'house' else 0
    data['type_studio'] = 1 if selected_type == 'studio' else 0

    # One-hot for city
    selected_city = data.pop('city', 'Other_City')
    encoded_city = selected_city if selected_city in CITY_OPTIONS[:-1] else 'Other_City'

    # initialize OHE city columns (any column in feature_cols starting with 'city_encoded_')
    city_ohe_cols = [col for col in feature_cols if col.startswith('city_encoded_')]
    for col in city_ohe_cols:
        data[col] = 0
    ohe_col_name = f'city_encoded_{encoded_city}'
    if ohe_col_name in city_ohe_cols:
        data[ohe_col_name] = 1

    # Make DataFrame and align columns
    final_df = pd.DataFrame([data])
    final_df = final_df.reindex(columns=feature_cols, fill_value=0)

    return final_df, analysis_flags

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.image("Victor.jpg", width=120)
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.radio("", options=["Prediction", "Info"])

# ---------------------------
# Info page (moved out of sidebar)
# ---------------------------
if page == "Info":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex; align-items:center; justify-content:space-between'>", unsafe_allow_html=True)
    st.markdown("<h1 class='big-title'>🏷️ Project Information</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='muted'>
        This application predicts **weekly** rental prices for student accommodation using a trained ML model.
        The model expects preprocessed and scaled features (one-hot encoded cities/types, engineered features for distances and deposit ratios).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Project metadata
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Objective")
        st.write("Predict **monthly** rental price (GBP) for student-focused properties in UK cities.")
        st.subheader("Model")
        st.write(f"Core model: **{model_name}**")
        with st.expander("Model Performance "):
            st.write("- R²: 0.8349 ")
            st.write("- RMSE (log-target): 0.1347")

    with col2:
        st.subheader("Data & Preprocessing")
        st.write(
            "- Standard scaling and one-hot encoding\n- Engineered distance categories (uni / station)\n- Amenities & tenancy flags"
        )
        

    st.markdown("---")
    st.subheader("Team / Contact")
    st.write("* Student: [Your Name/ID]  \n* Supervisor: [Supervisor Name]  \n* Sponsor: PTDF (example)")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------------------------
# Prediction page
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h1 class='big-title'>🏡 UK Student Rental Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Enter property and location details to get a predicted <strong>monthly</strong> rental estimate.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("")

# Prediction form organized into sections
with st.form("rental_prediction_form", clear_on_submit=False):
    st.markdown("### <span class='section-title'>1. Property Details</span>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2, step=1)
        rooms_available = st.number_input("Rooms Available", min_value=1, max_value=10, value=2, step=1)
    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=1, step=1)
        prop_type = st.selectbox("Property Type", options=PROPERTY_TYPES, index=0)
    with col3:
        deposit = st.number_input("Deposit (£)", min_value=0, value=500, step=10)
        # optional: allow input of monthly rent if known (not used by model)
        known_rent = st.number_input("Known Weekly Rent (optional)", min_value=0.0, value=0.0, step=10.0)

    st.markdown("---")
    st.markdown("### <span class='section-title'>2. Location & Distances</span>", unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        city = st.selectbox("City", options=CITY_OPTIONS, index=1)
    with col5:
        dist_uni = st.number_input("Distance to Closest University (km)", min_value=0.0, value=1.5, step=0.1)
    with col6:
        dist_station = st.number_input("Distance to Closest Train Station (km)", min_value=0.0, value=0.8, step=0.1)

    st.markdown("---")
    st.markdown("### <span class='section-title'>3. Amenities & Certifications</span>", unsafe_allow_html=True)
    # Use expanders for neatness
    amenity_inputs = {}
    with st.expander("🔑 Essential Amenities", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        amenity_inputs['Furnished'] = c1.checkbox("Furnished", value=True)
        amenity_inputs['Wifi'] = c2.checkbox("Dedicated Wifi", value=True)
        amenity_inputs['Internet'] = c3.checkbox("General Internet", value=True)
        amenity_inputs['Gas'] = c4.checkbox("Gas Included", value=True)
        # additional row
        c5, c6, c7, c8 = st.columns(4)
        amenity_inputs['Electricity'] = c5.checkbox("Electricity Included", value=True)
        amenity_inputs['Water'] = c6.checkbox("Water Included", value=True)
        amenity_inputs['Double glazing'] = c7.checkbox("Double Glazing", value=True)
        amenity_inputs['Parking'] = c8.checkbox("Parking", value=False)

    with st.expander("🍽️ Kitchen Appliances", expanded=False):
        k1, k2, k3 = st.columns(3)
        amenity_inputs['Fridge'] = k1.checkbox("Fridge", value=True)
        amenity_inputs['Freezer'] = k1.checkbox("Freezer", value=True)
        amenity_inputs['Oven'] = k2.checkbox("Oven", value=True)
        amenity_inputs['Microwave'] = k2.checkbox("Microwave", value=True)
        amenity_inputs['Dishwasher'] = k3.checkbox("Dishwasher", value=False)
        amenity_inputs['Washing machine'] = k3.checkbox("Washing Machine", value=True)

    with st.expander("🛠️ Safety & Certification", expanded=False):
        s1, s2, s3 = st.columns(3)
        amenity_inputs['Fire alarm'] = s1.checkbox("Fire Alarm", value=True)
        amenity_inputs['Alarm'] = s1.checkbox("Security Alarm", value=False)
        amenity_inputs['EPC'] = s2.checkbox("EPC", value=True)
        amenity_inputs['Gas Safe'] = s2.checkbox("Gas Safe Reg.", value=True)
        amenity_inputs['Elec Cert'] = s3.checkbox("Elec. Safety Cert.", value=True)
        amenity_inputs['NRLA'] = s3.checkbox("NRLA Registered", value=False)

    st.markdown("")
    submitted = st.form_submit_button("🔍 Predict Weekly Rent")

# Prediction logic
if submitted:
    # Collect raw input
    raw_input = {
        'Bedrooms': int(bedrooms),
        'Bathrooms': int(bathrooms),
        'rooms_available': int(rooms_available),
        'deposit': float(deposit),
        'type': prop_type,
        'city': city,
        'distance_to_uni_km': float(dist_uni),
        'distance_to_station_km': float(dist_station),
    }
    # Add amenity flags
    for key, value in amenity_inputs.items():
        raw_input[key] = 1 if value else 0

    try:
        # Feature engineering
        features_df, flags = engineer_features(raw_input, feature_columns, uni_distance_mapping, station_distance_mapping)

        # Scale using loaded scaler (assumes scaler expects same columns/order)
        scaled = scaler.transform(features_df.values)  # shape (1, n_features)

        # Predict (model returns log-target in your original code)
        log_price_pred = model.predict(scaled)[0]
        predicted_monthly_price = float(np.exp(log_price_pred))

        # Display result - prominent
        st.markdown("---")
        st.subheader("💡 Prediction Result")
        # Use columns to center the result visually
        a, b, c = st.columns([1, 2, 1])
        with b:
            st.markdown(f"<div style='text-align:center; padding: 1rem; border-radius: 12px; background: linear-gradient(90deg,#fff7ed,#ffffff);'>"
                        f"<h2 style='margin:0'>£{predicted_monthly_price:,.2f}</h2>"
                        f"<div class='muted'>Estimated Weekly Rent</div></div>", unsafe_allow_html=True)
        st.markdown("")

        # Smart analysis panel
        st.markdown("### 📊 Smart Feature Analysis")
        left, right = st.columns(2)
        uni_cat = pd.cut([dist_uni], bins=UNI_BINS, labels=UNI_LABELS, right=True)[0]
        station_cat = pd.cut([dist_station], bins=STATION_BINS, labels=STATION_LABELS, right=True)[0]

        with left:
            st.info(
                f"**Location Insights**\n\n• **University Proximity:** {uni_cat} ({dist_uni} km)\n"
                f"• **Transport Access:** {station_cat} ({dist_station} km)\n"
            )
        with right:
            tenancy = "Likely shared tenancy" if flags.get('Is_Shared_Tenancy', 0) == 1 else "Standard let"
            deposit_msg = "Zero deposit (attractive to students)" if flags.get('Zero_Deposit_Required', 0) == 1 else "Standard deposit required"
            st.info(
                f"**Property Insights**\n\n• **Tenancy Type:** {tenancy} ({rooms_available}/{bedrooms})\n"
                f"• **Deposit:** {deposit_msg} (£{deposit})\n"
                f"• **Key Amenities:** {'Furnished' if amenity_inputs.get('Furnished') else 'Unfurnished'}, "
                f"{'Wifi' if amenity_inputs.get('Wifi') else 'No Wifi'}\n"
            )

        # Optional: show the top few feature values used for prediction
        with st.expander("See engineered & input features used for prediction"):
            display_df = features_df.T.rename(columns={0: "value"})
            st.dataframe(display_df.style.format("{:.3g}"), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.exception(e)

# Footer / small note
st.markdown("---")
st.markdown("<div class='muted'>Tip: If your saved model was trained on weekly prices, convert target consistently before saving/loading the model. This app assumes the model's prediction is a log-transformed monthly price and therefore uses exp() to inverse-transform.</div>", unsafe_allow_html=True)
