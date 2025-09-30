"""
Streamlit demo application for mental health tweet classification.

⚠️ IMPORTANT: This is for research and demonstration purposes ONLY.
NOT for clinical diagnosis or medical advice.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import time
from typing import Optional, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our modules with error handling
try:
    from src.models.baseline import BaselineModel
    from src.data.preprocess import TweetPreprocessor
    BASELINE_AVAILABLE = True
except ImportError as e:
    BASELINE_AVAILABLE = False
    st.error(f"Baseline model not available: {e}")

try:
    from src.models.transformer import TransformerModel
    TRANSFORMER_AVAILABLE = True
except ImportError as e:
    TRANSFORMER_AVAILABLE = False
    st.warning(f"Transformer model not available: {e}")

# Crisis resources
CRISIS_RESOURCES = {
    "🇺🇸 US National Suicide Prevention Lifeline": "988",
    "🇺🇸 Crisis Text Line": "Text HOME to 741741", 
    "🇬🇧 UK Samaritans": "116 123",
    "🌍 International Crisis Lines": "https://www.iasp.info/resources/Crisis_Centres/"
}

def load_model(model_type: str, model_path: str) -> Optional[Any]:
    """Load a trained model."""
    try:
        if model_type == "baseline" and BASELINE_AVAILABLE:
            model = BaselineModel()
            model.load(model_path)
            return model
        elif model_type == "transformer" and TRANSFORMER_AVAILABLE:
            model = TransformerModel.load_model(model_path)
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading {model_type} model: {e}")
        return None

def show_crisis_resources():
    """Display crisis resources prominently."""
    st.error("""
    🚨 **IF YOU ARE IN CRISIS, PLEASE SEEK IMMEDIATE HELP**
    
    This tool is NOT for crisis intervention or medical diagnosis.
    """)
    
    with st.expander("🆘 Crisis Resources - Click Here for Help", expanded=False):
        st.markdown("### Immediate Help Resources:")
        
        for resource, contact in CRISIS_RESOURCES.items():
            st.markdown(f"**{resource}**: {contact}")
        
        st.markdown("""
        ### Additional Resources:
        - **Emergency Services**: Call 911 (US), 999 (UK), or your local emergency number
        - **Crisis Text Line**: https://www.crisistextline.org/
        - **National Alliance on Mental Illness**: https://www.nami.org/
        - **Mental Health America**: https://www.mhanational.org/
        """)

def show_disclaimers():
    """Display important disclaimers."""
    st.markdown("""
    ---
    ### ⚠️ Important Disclaimers
    
    **🏥 NOT FOR MEDICAL USE**
    - This tool is for research and demonstration purposes ONLY
    - NOT intended for clinical diagnosis, medical advice, or treatment
    - Results should NOT be used to make medical decisions
    
    **🎯 RESEARCH PURPOSE**
    - This is an experimental AI model for academic research
    - Accuracy is not guaranteed and may vary significantly
    - May not represent all populations or contexts
    
    **👥 SEEK PROFESSIONAL HELP**
    - Always consult qualified mental health professionals
    - This tool cannot replace human judgment or professional assessment
    - If concerned about mental health, contact a healthcare provider
    
    **🔒 PRIVACY**
    - Text you enter is processed but not stored permanently
    - Do not enter personally identifiable information
    - This demo is not HIPAA compliant
    
    ---
    """)

def preprocess_text(text: str) -> str:
    """Preprocess input text."""
    try:
        preprocessor = TweetPreprocessor(
            remove_urls=True,
            remove_mentions=False,
            expand_contractions=True,
            lowercase=True
        )
        return preprocessor.clean_text(text)
    except Exception:
        # Fallback to basic cleaning
        return text.strip().lower()

def format_prediction_result(prediction: int, 
                           probability: float, 
                           text: str,
                           model_type: str) -> Dict[str, Any]:
    """Format prediction results for display."""
    
    class_names = {0: "Non-Depression", 1: "Depression"}
    confidence = max(probability, 1 - probability) * 100
    
    result = {
        "predicted_class": class_names[prediction],
        "confidence": confidence,
        "probability_depression": probability * 100 if prediction == 1 else (1 - probability) * 100,
        "probability_non_depression": (1 - probability) * 100 if prediction == 1 else probability * 100,
        "model_type": model_type,
        "text_length": len(text),
        "word_count": len(text.split())
    }
    
    return result

def main():
    """Main Streamlit application."""
    
    # Page config
    st.set_page_config(
        page_title="Mental Health Tweet Classifier - Research Demo",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5em;
        padding: 1em;
        margin: 1em 0;
    }
    .crisis-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5em;
        padding: 1em;
        margin: 1em 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-header">🧠 Mental Health Tweet Classifier</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2em; color: #666;">Research Demo - Depression Detection in Social Media Text</p>', 
                unsafe_allow_html=True)
    
    # Show crisis resources prominently
    show_crisis_resources()
    
    # Sidebar
    st.sidebar.title("🔧 Model Configuration")
    
    # Model selection
    available_models = []
    if BASELINE_AVAILABLE:
        available_models.append("Baseline (TF-IDF + Logistic Regression)")
    if TRANSFORMER_AVAILABLE:
        available_models.append("Transformer (RoBERTa)")
    
    if not available_models:
        st.error("No models available. Please ensure models are trained and accessible.")
        st.stop()
    
    selected_model_display = st.sidebar.selectbox(
        "Choose Model Type:",
        available_models,
        index=0
    )
    
    # Map display name to internal name
    if "Baseline" in selected_model_display:
        model_type = "baseline"
        default_path = "models/baseline"
    else:
        model_type = "transformer"  
        default_path = "models/transformer"
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path:",
        value=default_path,
        help="Path to the trained model directory"
    )
    
    # Load model
    with st.spinner(f"Loading {model_type} model..."):
        model = load_model(model_type, model_path)
    
    if model is None:
        st.error(f"Could not load {model_type} model from {model_path}")
        st.info("Please ensure you have trained a model first using the training scripts.")
        st.stop()
    
    st.sidebar.success(f"✅ {model_type.title()} model loaded successfully!")
    
    # Model info
    with st.sidebar.expander("ℹ️ Model Information"):
        if hasattr(model, 'training_stats'):
            st.json(model.training_stats)
        else:
            st.write(f"Model Type: {model_type}")
            st.write("Training stats not available")
    
    # Main interface
    st.markdown("## 📝 Enter Text for Analysis")
    
    # Text input methods
    input_method = st.radio(
        "Input method:",
        ["Type text", "Upload file", "Use examples"],
        horizontal=True
    )
    
    user_text = ""
    
    if input_method == "Type text":
        user_text = st.text_area(
            "Enter tweet or social media text:",
            height=100,
            placeholder="Enter text here... (e.g., 'Feeling really down today, nothing seems to matter anymore')",
            help="Enter any social media text for analysis. Avoid sharing personal identifying information."
        )
        
    elif input_method == "Upload file":
        uploaded_file = st.file_uploader(
            "Upload a text file:",
            type=['txt'],
            help="Upload a .txt file containing the text to analyze"
        )
        
        if uploaded_file is not None:
            user_text = str(uploaded_file.read(), "utf-8")
            st.text_area("Uploaded text:", user_text, height=100, disabled=True)
    
    elif input_method == "Use examples":
        example_texts = {
            "Depression example": "I feel so hopeless and empty inside. Nothing I do matters anymore and I just want to disappear.",
            "Non-depression example": "Just finished an amazing workout! Feeling energized and ready to tackle the day. Life is good!",
            "Neutral example": "Heading to the grocery store to pick up some ingredients for dinner tonight.",
            "Ambiguous example": "Another long day at work. Sometimes I wonder if this is all there is to life."
        }
        
        selected_example = st.selectbox("Choose an example:", list(example_texts.keys()))
        user_text = example_texts[selected_example]
        st.text_area("Selected example:", user_text, height=100, disabled=True)
    
    # Analysis button
    if st.button("🔍 Analyze Text", type="primary", disabled=not user_text.strip()):
        
        if len(user_text.strip()) < 5:
            st.warning("Please enter at least 5 characters for analysis.")
            return
        
        # Show analysis in progress
        with st.spinner("Analyzing text... This may take a moment."):
            
            try:
                # Preprocess text
                processed_text = preprocess_text(user_text)
                
                # Make prediction
                if model_type == "baseline":
                    prediction = model.predict([processed_text])[0]
                    probabilities = model.predict_proba([processed_text])[0]
                    prob_depression = probabilities[1]
                else:
                    predictions, probabilities = model.predict([processed_text])
                    prediction = predictions[0]
                    prob_depression = probabilities[0][1]
                
                # Format results
                result = format_prediction_result(
                    prediction, prob_depression, user_text, model_type
                )
                
                # Display results
                st.markdown("## 📊 Analysis Results")
                
                # Create columns for results
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Main prediction
                    if result["predicted_class"] == "Depression":
                        st.error(f"🔴 **Prediction: {result['predicted_class']}**")
                        st.markdown(f"**Confidence:** {result['confidence']:.1f}%")
                    else:
                        st.success(f"🟢 **Prediction: {result['predicted_class']}**")
                        st.markdown(f"**Confidence:** {result['confidence']:.1f}%")
                
                with col2:
                    st.metric(
                        "Depression Likelihood", 
                        f"{result['probability_depression']:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Non-Depression Likelihood",
                        f"{result['probability_non_depression']:.1f}%"
                    )
                
                # Detailed results
                with st.expander("📋 Detailed Results"):
                    st.json(result)
                
                # Interpretation help
                st.markdown("### 🤔 How to Interpret Results")
                
                if result["confidence"] >= 80:
                    confidence_text = "High confidence"
                    confidence_color = "🔵"
                elif result["confidence"] >= 60:
                    confidence_text = "Moderate confidence" 
                    confidence_color = "🟡"
                else:
                    confidence_text = "Low confidence"
                    confidence_color = "🟠"
                
                st.markdown(f"""
                {confidence_color} **{confidence_text}** prediction ({result['confidence']:.1f}%)
                
                **What this means:**
                - The model predicts this text is **{result['predicted_class'].lower()}**-related
                - Confidence indicates how certain the model is about this prediction
                - Higher confidence does NOT mean medical accuracy
                
                **Remember:** This is an experimental AI model and should not be used for medical decisions.
                """)
                
                # Show crisis resources if depression detected
                if result["predicted_class"] == "Depression":
                    st.markdown("---")
                    st.markdown("### 🆘 If You're Struggling")
                    st.error("""
                    If this text reflects how you're feeling, please remember:
                    - You are not alone
                    - Help is available 24/7
                    - Speaking with a professional can make a difference
                    """)
                    
                    # Show crisis resources again
                    for resource, contact in CRISIS_RESOURCES.items():
                        st.markdown(f"**{resource}**: {contact}")
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.info("Please try again or contact support if the issue persists.")
    
    # Show disclaimers at the bottom
    show_disclaimers()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>Mental Health Tweet Classifier Research Demo | 
    <a href='#' onclick='window.scrollTo(0,0)'>🆘 Crisis Resources</a> | 
    Built for educational purposes</p>
    <p><strong>Remember:</strong> This tool is for research only. Always consult mental health professionals for support.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()