import streamlit as st
import joblib
import numpy as np

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Retinopathy Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ------------------ LOAD ------------------
model = joblib.load("retinopathy_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------ CSS ------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f4f6f9;
}

/* Header Bar */
.header-bar {
    background-color: #1f2937;
    padding: 15px 25px;
    border-radius: 8px;
    color: white;
    font-size: 20px;
    font-weight: 600;
}

/* Section Box */
.section-box {
    background: white;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    margin-top: 15px;
}

/* Labels */
.section-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Button */
.stButton>button {
    background-color: #111827;
    color: white;
    border-radius: 6px;
    height: 42px;
    font-weight: 500;
}

/* Result */
.result-box {
    padding: 18px;
    border-radius: 8px;
    margin-top: 15px;
}

.success {
    background-color: #e6f4ea;
    color: #166534;
}

.error {
    background-color: #fdecea;
    color: #991b1b;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="header-bar">🩺 Clinical Risk Prediction Dashboard</div>', unsafe_allow_html=True)

# ------------------ LAYOUT ------------------
col1, col2 = st.columns([2, 1])

# ------------------ LEFT SIDE (FORM) ------------------
with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Parameters</div>', unsafe_allow_html=True)

    age = st.number_input("Age", 1.0, 120.0, 60.0)
    systolic_bp = st.number_input("Systolic BP", 50.0, 200.0, 120.0)
    diastolic_bp = st.number_input("Diastolic BP", 40.0, 150.0, 80.0)
    cholesterol = st.number_input("Cholesterol", 50.0, 300.0, 150.0)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ RIGHT SIDE (ACTION + RESULT) ------------------
with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Assessment</div>', unsafe_allow_html=True)

    if st.button("Run Prediction", use_container_width=True):

        input_data = np.array([[age, systolic_bp, diastolic_bp, cholesterol]])
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)

        prob = probability[0][1]

        st.progress(int(prob * 100))

        if prediction[0] == 1:
            st.markdown(
                f'<div class="result-box error"><b>High Risk</b><br>Probability: {prob:.2%}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box success"><b>Low Risk</b><br>Probability: {(1-prob):.2%}</div>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Clinical decision support tool. Not a substitute for professional medical advice.")
