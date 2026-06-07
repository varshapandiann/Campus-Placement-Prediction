import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🏢",
    layout="centered"
)

# Load and prepare model
@st.cache_resource
def load_model():
    # Load dataset
    df = pd.read_csv("Placement_Data_Full_Class.csv")
    
    # Drop irrelevant columns if any (like 'sl_no', 'salary')
    if 'sl_no' in df.columns:
        df = df.drop(columns=['sl_no'])
    if 'salary' in df.columns:
        df = df.drop(columns=['salary'])
    
    # Encode categorical columns
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # Split features and target
    X = df.drop(columns=["status"])   # features
    y = df["status"]                  # target (placed or not)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X.columns.tolist(), label_encoders

model, feature_names, label_encoders = load_model()

# Streamlit UI
st.title("Placement Prediction App 👩🏻‍🎓🎓")
st.write("Enter student details below to predict placement status.")

# User input form
with st.form("prediction_form"):
    user_input = {}
    for feature in feature_names:
        # If categorical, show selectbox
        if feature in label_encoders:
            options = label_encoders[feature].classes_
            user_input[feature] = st.selectbox(f"{feature}", options)
        else:
            user_input[feature] = st.number_input(f"{feature}", value=0.0)
    
    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    # Convert user input to dataframe
    input_df = pd.DataFrame([user_input])
    
    # Encode categorical fields
    for feature, le in label_encoders.items():
        if feature in input_df.columns:
            input_df[feature] = le.transform(input_df[feature])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Decode prediction back to original labels
    target_le = LabelEncoder()
    target_le.fit(["Not Placed", "Placed"])  # target values in dataset
    prediction_label = target_le.inverse_transform([prediction])[0]
    
    st.success(f"Prediction: {prediction_label}")