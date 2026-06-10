import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Campus Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Campus Placement Prediction")
st.write(
    "Enter the candidate details below to predict whether the student is likely to be placed."
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return model, label_encoders


model, label_encoders = load_artifacts()

MODEL_FEATURES = list(model.feature_names_in_)

CATEGORICAL_FEATURES = [
    feature
    for feature in MODEL_FEATURES
    if feature in label_encoders and feature != "status"
]

FRIENDLY_NAMES = {
    "gender": "Gender",
    "ssc_p": "SSC Percentage",
    "ssc_b": "SSC Board",
    "hsc_p": "HSC Percentage",
    "hsc_b": "HSC Board",
    "hsc_s": "HSC Stream",
    "degree_p": "Degree Percentage",
    "degree_t": "Degree Type",
    "workex": "Work Experience",
    "etest_p": "Employability Test Percentage",
    "specialisation": "MBA Specialisation",
    "mba_p": "MBA Percentage",
}

with st.form("prediction_form"):
    user_inputs = {}
    for feature in MODEL_FEATURES:
        label = FRIENDLY_NAMES.get(
            feature,
            feature.replace("_", " ").title()
        )

        if feature in CATEGORICAL_FEATURES:
            encoder = label_encoders[feature]
            selected_value = st.selectbox(
                label,
                options=list(encoder.classes_)
            )

            user_inputs[feature] = encoder.transform(
                [selected_value]
            )[0]

        else:
            user_inputs[feature] = st.number_input(
                label,
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=0.1
            )

    submitted = st.form_submit_button("Predict Placement")

if submitted:
    input_df = pd.DataFrame(
        [[user_inputs[col] for col in MODEL_FEATURES]],
        columns=MODEL_FEATURES
    )

    prediction = model.predict(input_df)[0]

    status_encoder = label_encoders["status"]
    predicted_label = status_encoder.inverse_transform(
        [prediction]
    )[0]

    st.divider()
    if predicted_label.lower() == "placed":
        st.success(
            "🎉 Prediction: The student is likely to be **Placed**."
        )
    else:
        st.error(
            "⚠️ Prediction: The student is likely to be **Not Placed**."
        )
