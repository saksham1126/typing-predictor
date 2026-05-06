import streamlit as st
import pandas as pd
import joblib
import os
import time

# Set page config
st.set_page_config(
    page_title="Typing Efficiency Predictor",
    page_icon="⌨️",
    layout="centered"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #1f77b4;
    }
    .ref-text {
        font-size: 18px;
        background-color: rgba(128, 128, 128, 0.15);
        border: 1px solid rgba(128, 128, 128, 0.3);
        color: inherit;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⌨️ Typing Speed vs Accuracy Predictor")
st.markdown("""
    Welcome to the **Typing Efficiency Predictor**! 
    This app uses Machine Learning to calculate your true typing efficiency score based on your raw speed, accuracy, and error rate.
""")

@st.cache_resource
def load_model():
    model_path = "models/best_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("⚠️ Model not found! Please run `python src/model_training.py` first to train and save the model.")
    st.stop()

tab1, tab2 = st.tabs(["🚀 Live Typing Test", "🎛️ Manual Input"])

with tab1:
    st.header("📝 Take a Typing Test")
    
    reference_text = "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. By training algorithms on large datasets, these models can identify patterns, make decisions, and improve their performance over time without being explicitly programmed."
    
    st.markdown('<div class="ref-text">' + reference_text + '</div>', unsafe_allow_html=True)
    
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'test_active' not in st.session_state:
        st.session_state.test_active = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Test"):
            st.session_state.start_time = time.time()
            st.session_state.test_active = True
            st.rerun()
            
    with col2:
        if st.button("Reset"):
            st.session_state.start_time = None
            st.session_state.test_active = False
            st.rerun()

    if st.session_state.test_active:
        st.info("Test is running! Start typing in the box below and click 'Finish & Calculate' when done.")
        user_input = st.text_area("Type the text here:", height=150)
        
        if st.button("Finish & Calculate"):
            end_time = time.time()
            time_taken_sec = end_time - st.session_state.start_time
            time_taken_min = time_taken_sec / 60.0
            
            # Calculations
            typed_words = user_input.split()
            ref_words = reference_text.split()
            
            # WPM (Standard definition: 5 chars = 1 word)
            gross_wpm = (len(user_input) / 5.0) / time_taken_min if time_taken_min > 0 else 0
            
            # Errors
            errors = 0
            for i in range(max(len(ref_words), len(typed_words))):
                if i < len(ref_words) and i < len(typed_words):
                    if ref_words[i] != typed_words[i]:
                        errors += 1
                else:
                    errors += 1
                    
            errors_per_min = errors / time_taken_min if time_taken_min > 0 else 0
            
            # Accuracy
            total_words = max(len(ref_words), len(typed_words))
            accuracy = max(0, ((total_words - errors) / total_words) * 100) if total_words > 0 else 0
            
            # Feature engineering
            consistency_score = accuracy / (errors_per_min + 1)
            
            input_data = pd.DataFrame({
                'speed_wpm': [gross_wpm],
                'accuracy_percent': [accuracy],
                'error_rate': [errors_per_min],
                'consistency_score': [consistency_score]
            })
            
            prediction = model.predict(input_data)[0]
            prediction = max(0, min(100, prediction))
            
            st.success(f"Test Completed in {time_taken_sec:.1f} seconds!")
            
            # Display stats
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Speed (WPM)", f"{gross_wpm:.1f}")
            col_b.metric("Accuracy", f"{accuracy:.1f}%")
            col_c.metric("Errors/Min", f"{errors_per_min:.1f}")
            
            st.markdown("---")
            st.markdown(f'<p class="big-font">🏆 Estimated Efficiency Score: {prediction:.2f} / 100</p>', unsafe_allow_html=True)
            
            if prediction >= 80:
                st.info("Excellent! You maintain a great balance of high speed and high accuracy.")
            elif prediction >= 50:
                st.warning("Good, but there's room for improvement. Try to focus on accuracy first, then speed.")
            else:
                st.error("Needs Improvement. High error rates or low accuracy significantly bring down your true typing efficiency.")
            
            # Stop test state
            st.session_state.test_active = False
            st.session_state.start_time = None

with tab2:
    st.header("📊 Enter Your Typing Metrics")

    col1, col2 = st.columns(2)

    with col1:
        speed = st.slider("Typing Speed (WPM)", min_value=10, max_value=150, value=60, step=1)
        accuracy_val = st.slider("Accuracy (%)", min_value=50.0, max_value=100.0, value=95.0, step=0.5)

    with col2:
        error_rate = st.slider("Errors per Minute", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
        
    consistency_score = accuracy_val / (error_rate + 1)

    input_data = pd.DataFrame({
        'speed_wpm': [speed],
        'accuracy_percent': [accuracy_val],
        'error_rate': [error_rate],
        'consistency_score': [consistency_score]
    })

    if st.button("Predict Efficiency Score"):
        with st.spinner("Calculating..."):
            prediction = model.predict(input_data)[0]
            prediction = max(0, min(100, prediction))
            
            st.success("Calculation Complete!")
            st.markdown(f'<p class="big-font">🏆 Estimated Efficiency Score: {prediction:.2f} / 100</p>', unsafe_allow_html=True)
            
            st.subheader("💡 Interpretation")
            if prediction >= 80:
                st.info("Excellent! You maintain a great balance of high speed and high accuracy.")
            elif prediction >= 50:
                st.warning("Good, but there's room for improvement. Try to focus on accuracy first, then speed.")
            else:
                st.error("Needs Improvement. High error rates or low accuracy significantly bring down your true typing efficiency.")

st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit & Scikit-Learn*")
