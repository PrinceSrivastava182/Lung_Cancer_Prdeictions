# Lung Cancer Prediction Web App with Enhanced Visualizations
# Corrected version with all 15 features

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Risk Assessment",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler with error handling
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load('random_forest_best.pkl')
        _, _, _, _, scaler = joblib.load('data_preprocessed.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        st.stop()

model, scaler = load_model_assets()

# Main header
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 2.8em;'>🫁 Lung Cancer Risk Assessment</h1>
        <p style='font-size: 1.2em; color: #666;'>A predictive analytics tool for early risk detection</p>
    </div>
""", unsafe_allow_html=True)

# Medical disclaimer
st.warning("""
**Disclaimer**: This tool provides statistical risk estimates only and is not a diagnostic device. 
Clinical evaluation by a qualified healthcare provider is essential for accurate diagnosis.
""")

st.markdown("---")

# Sidebar Inputs
st.sidebar.header("Patient Health Profile")

def user_input_features():
    # Add help expander
    with st.sidebar.expander("ℹ️ About These Questions"):
        st.write("""
        These questions help assess common risk factors associated with lung conditions.
        Please answer based on your current health status.
        """)

    # Demographic
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age (years)", 18, 100, 55)

    # Lifestyle factors
    smoking = st.sidebar.selectbox("Do you smoke?", ["Yes", "No"], 
                                 help="Includes cigarettes, cigars, or vaping")
    alcohol_consuming = st.sidebar.selectbox("Regular alcohol consumption?", ["Yes", "No"])
    peer_pressure = st.sidebar.selectbox("History of peer pressure to smoke?", ["Yes", "No"])

    # Symptoms
    yellow_fingers = st.sidebar.selectbox("Yellowish fingers?", ["Yes", "No"])
    anxiety = st.sidebar.selectbox("Frequent anxiety?", ["Yes", "No"])
    chronic_disease = st.sidebar.selectbox("Chronic lung disease?", ["Yes", "No"])
    fatigue = st.sidebar.selectbox("Persistent fatigue?", ["Yes", "No"])
    allergy = st.sidebar.selectbox("Respiratory allergies?", ["Yes", "No"])
    wheezing = st.sidebar.selectbox("Wheezing sounds when breathing?", ["Yes", "No"])
    coughing = st.sidebar.selectbox("Chronic cough?", ["Yes", "No"])
    shortness_of_breath = st.sidebar.selectbox("Shortness of breath?", ["Yes", "No"])
    swallowing_difficulty = st.sidebar.selectbox("Difficulty swallowing?", ["Yes", "No"])
    chest_pain = st.sidebar.selectbox("Chest pain?", ["Yes", "No"])

    # Prepare data in EXACT SAME ORDER as training data
    data = [
        1 if gender == "Male" else 0,  # Gender
        age,                          # Age
        1 if smoking == "Yes" else 0,  # Smoking
        1 if yellow_fingers == "Yes" else 0,
        1 if anxiety == "Yes" else 0,
        1 if peer_pressure == "Yes" else 0,
        1 if chronic_disease == "Yes" else 0,
        1 if fatigue == "Yes" else 0,
        1 if allergy == "Yes" else 0,
        1 if wheezing == "Yes" else 0,
        1 if alcohol_consuming == "Yes" else 0,
        1 if coughing == "Yes" else 0,
        1 if shortness_of_breath == "Yes" else 0,
        1 if swallowing_difficulty == "Yes" else 0,
        1 if chest_pain == "Yes" else 0
    ]

    return np.array(data).reshape(1, -1), data, age, smoking

input_data, raw_data, age, smoking = user_input_features()

# Feature names in correct order
feature_names = [
    "Gender", "Age", "Smoking", "Yellow Fingers", "Anxiety",
    "Peer Pressure", "Chronic Disease", "Fatigue", "Allergy",
    "Wheezing", "Alcohol", "Coughing", "Shortness of Breath",
    "Swallowing Difficulty", "Chest Pain"
]

# Prediction button
if st.button("🔍 Assess My Risk", type="primary", use_container_width=True):
    with st.spinner('Analyzing your health profile...'):
        time.sleep(1.5)

        # Scale and predict
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probabilities = model.predict_proba(scaled_data)[0]
        risk_percentage = probabilities[1] * 100

        # Create results dataframe
        results_df = pd.DataFrame([raw_data], columns=feature_names)
        results_df['Risk Percentage'] = risk_percentage
        results_df['Risk Category'] = "High" if prediction == 1 else "Low"

        # ========== VISUALIZATION 1: RISK GAUGE ==========
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Your Lung Cancer Risk Score"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.8,
                    'value': risk_percentage}
            }))

        fig_gauge.update_layout(height=300, margin=dict(t=50, b=10))

        # ========== VISUALIZATION 2: FEATURE IMPORTANCE ==========
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='reds',
            title="Which Factors Contribute Most to Your Risk"
        )
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})

        # ========== VISUALIZATION 3: AGE-SMOKING RISK HEATMAP ==========
        # Simulated data for demonstration
        age_groups = ["<30", "30-39", "40-49", "50-59", "60+"]
        smoking_status = ["Non-smoker", "Former smoker", "Current smoker"]
        risk_data = [
            [5, 15, 35],  # <30
            [8, 25, 45],   # 30-39
            [15, 35, 60],  # 40-49
            [25, 50, 75],  # 50-59
            [35, 65, 85]   # 60+
        ]

        fig_heatmap = px.imshow(
            risk_data,
            labels=dict(x="Smoking Status", y="Age Group", color="Risk %"),
            x=smoking_status,
            y=age_groups,
            color_continuous_scale="reds",
            aspect="auto",
            text_auto=".0f",
            title="Population Risk by Age and Smoking Status"
        )
        fig_heatmap.update_xaxes(side="top")

        # Display results
        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(fig_gauge, use_container_width=True)
            if prediction == 1:
                st.error(f"**High Risk Detected** ({risk_percentage:.1f}%) - Consider consulting a pulmonologist")
            else:
                st.success(f"**Low Risk Identified** ({risk_percentage:.1f}%) - Maintain healthy habits")

            # Export results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Report",
                data=csv,
                file_name="lung_health_assessment.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.caption("Comparison against general population risk estimates")

        st.markdown("---")

        # Feature importance and raw data tabs + Recommendations
        tab1, tab2, tab3 = st.tabs(["📈 Risk Factor Analysis", "📋 Your Health Profile", "💡 Recommendations"])

        with tab1:
            st.plotly_chart(fig_importance, use_container_width=True)
            st.write("""
            **How to interpret this chart:**
            - Features higher on the list have greater impact on your risk score
            - Red bars indicate stronger influence on the prediction
            """)

        with tab2:
            st.dataframe(
                results_df.drop(columns=['Risk Percentage', 'Risk Category']).T.rename(columns={0: 'Your Response'}),
                use_container_width=True
            )
            st.write("""
            **Your responses:**  
            - 1 = Yes  
            - 0 = No  
            - Age in years
            """)

        with tab3:
            st.subheader("Personalized Health Tips")
            if prediction == 1:
                st.warning("Consider a lung CT scan and consultation with a pulmonologist.")
                if smoking == "Yes":
                    st.info("🚭 Quitting smoking can lower your risk dramatically within 1-2 years.")
                if age >= 50:
                    st.info("👴 Routine screenings are especially important after age 50.")
            else:
                st.success("👍 Keep up the healthy habits! Maintain regular checkups and avoid exposure to pollutants.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.9em; color: #666; line-height: 1.5;'>
    <p style='margin-bottom: 0.5em;'>This tool provides machine learning-based risk assessment only</p>
    <p style='margin-top: 0;'>© 2025 Prince Srivastava | Decision support system (non-clinical)</p>
</div>
""", unsafe_allow_html=True)
