"""
Simplified Streamlit demo for Mental Health Tweet Classifier.
"""

import streamlit as st
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import our simple model
try:
    from simple_model import SimpleModel
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# Crisis resources
CRISIS_RESOURCES = {
    "🇺🇸 US National Suicide Prevention Lifeline": "988",
    "🇺🇸 Crisis Text Line": "Text HOME to 741741", 
    "🇬🇧 UK Samaritans": "116 123",
    "🌍 International Crisis Lines": "https://www.iasp.info/resources/Crisis_Centres/"
}

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

def main():
    """Main Streamlit application."""
    
    # Page config
    st.set_page_config(
        page_title="Mental Health AI Research - Tweet Depression Classifier",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Modern CSS with glassmorphism effects and animations
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #000000 100%);
            background-attachment: fixed;
            color: #ffffff;
        }
        
        .main .block-container {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 2rem;
            margin-top: 1rem;
        }
        
        /* Modern buttons with dark glassmorphism */
        .stButton > button {
            background: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(20px) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 15px !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
        }
        
        .stButton > button:hover {
            background: rgba(102, 126, 234, 0.3) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4) !important;
            border: 1px solid rgba(102, 126, 234, 0.5) !important;
        }
        
        /* Text area with dark glassmorphism */
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(15px) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 15px !important;
            color: #ffffff !important;
        }
        
        .stTextArea textarea::placeholder {
            color: rgba(255, 255, 255, 0.6) !important;
        }
        
        /* Progress bars with modern styling */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
            border-radius: 10px !important;
        }
        
        /* Sidebar styling with dark theme */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.03) !important;
            backdrop-filter: blur(20px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.15) !important;
        }
        
        /* Dark theme text colors */
        .stMarkdown, .stText, p, span, div {
            color: #ffffff !important;
        }
        
        /* Sidebar text colors */
        .css-1d391kg .stMarkdown, .css-1d391kg p, .css-1d391kg span {
            color: #ffffff !important;
        }
        
        /* Modern font */
        * {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Smooth animations */
        * {
            transition: all 0.3s ease !important;
        }
        
        /* Enhanced text styling */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }
        
        /* Custom dark scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(102, 126, 234, 0.6);
        }
        
        /* Input field styling */
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.05) !important;
            color: #ffffff !important;
        }
        
        /* Metrics and other components */
        .metric-container {
            background: rgba(255, 255, 255, 0.05) !important;
            color: #ffffff !important;
        }
        
        /* Success and error messages for dark theme */
        .stSuccess, .stError, .stWarning, .stInfo {
            background: rgba(255, 255, 255, 0.08) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 15px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Dark Glassmorphism Header
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; 
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 50%, rgba(25, 167, 255, 0.2) 100%);
                backdrop-filter: blur(25px); border-radius: 25px; margin-bottom: 2rem; color: white;
                border: 1px solid rgba(255, 255, 255, 0.2); box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);">
        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700; 
                   background: linear-gradient(90deg, #ffffff 0%, #667eea 50%, #19a7ff 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
                   text-shadow: 0 0 20px rgba(102, 126, 234, 0.5);">
            🧠 Mental Health AI Research
        </h1>
        <h3 style="margin: 15px 0; font-weight: 300; color: #e2e8f0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
            Advanced Depression Detection in Social Media Text
        </h3>
        <p style="margin: 0; font-size: 1.2rem; color: #cbd5e0; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">
            Using Natural Language Processing and Machine Learning for Mental Health Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dark Glassmorphism Project Overview
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(20px); padding: 2rem; 
                border-radius: 20px; margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.15);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);">
        <h4 style="color: #ffffff; margin-top: 0; font-weight: 600; 
                   background: linear-gradient(135deg, #667eea 0%, #19a7ff 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            📋 About This Project
        </h4>
        <p style="color: #e2e8f0; margin-bottom: 0; line-height: 1.6; font-size: 1.05rem;">
            This AI-powered research tool analyzes social media text to identify potential indicators of depression. 
            Built using TF-IDF vectorization and Logistic Regression, it achieves <strong style="color: #667eea;">85.6% accuracy</strong> on our research dataset. 
            <strong style="background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%); 
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                This is a research demonstration only - not for medical diagnosis.
            </strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show crisis resources prominently
    show_crisis_resources()
    
    # Glassmorphism Sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1.5rem; 
               background: rgba(102, 126, 234, 0.15); backdrop-filter: blur(20px);
               border-radius: 20px; margin-bottom: 1.5rem;
               border: 1px solid rgba(255, 255, 255, 0.2);
               box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);">
        <h3 style="margin: 0; font-weight: 600;
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            ⚙️ AI Model Status
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    if MODEL_AVAILABLE:
        try:
            model = SimpleModel()
            if model.is_loaded():
                st.sidebar.success("🚀 AI Model Ready!")
                model_available = True
            else:
                st.sidebar.error("❌ Model files not found")
                model_available = False
        except Exception as e:
            st.sidebar.error(f"❌ Loading error: {e}")
            model_available = False
    else:
        st.sidebar.error("❌ Dependencies missing")
        model_available = False
    
    # Modern Model info
    with st.sidebar.expander("📊 Technical Specifications", expanded=model_available):
        if model_available:
            st.markdown("""
            **🤖 Algorithm:** Logistic Regression  
            **🔤 Features:** TF-IDF Vectors (10K dims)  
            **📚 Training:** ~8,305 social media posts  
            **🎯 Accuracy:** 85.6% (test set)  
            **📈 Performance:** Research-grade classifier
            """)
        else:
            st.warning("Model unavailable. Run the training notebook first.")
    
    # Quick stats
    if model_available:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Quick Stats")
        st.sidebar.info("""
        **Dataset Balance:**  
        • Depression: ~3,900 samples  
        • Non-Depression: ~4,400 samples
        
        **Model Features:**  
        • Balanced class weights  
        • N-gram analysis (1-2 grams)  
        • Stop word filtering
        """)
    
    # Main interface
    if not model_available:
        st.error("""
        **🚫 Model Not Available!** 
        
        To use this demo:
        1. Run the Jupyter notebook: `notebooks/01_explore.ipynb`
        2. Train the baseline model (this should create files in `models/baseline/`)
        3. Refresh this page
        """)
        st.info("💡 The notebook contains code to load data, train a model, and save it for use in this demo.")
        return
    
    # Dark Glassmorphism Analysis Section
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(25px); 
                padding: 2.5rem; border-radius: 25px; margin-bottom: 2rem;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4), 0 5px 15px rgba(0, 0, 0, 0.2);">
        <h3 style="color: #ffffff; margin-top: 0; display: flex; align-items: center; font-weight: 600;
                   background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            <span style="margin-right: 0.5rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));">🔍</span> 
            AI Text Analysis
        </h3>
    """, unsafe_allow_html=True)
    
    # Text input with modern styling
    user_text = st.text_area(
        "Enter social media text for analysis:",
        height=120,
        placeholder="Type or paste social media text here...\n\nExample: 'Feeling overwhelmed with everything going on in my life right now...'",
        help="💡 Enter any social media text for AI analysis. Please avoid sharing personal identifying information.",
        key="text_input"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Modern Analysis button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        analyze_clicked = st.button("� Analyze Text", type="primary", disabled=not user_text.strip(), use_container_width=True)
    
    if analyze_clicked:
        
        if len(user_text.strip()) < 5:
            st.warning("Please enter at least 5 characters for analysis.")
            return
        
        # Show analysis in progress
        with st.spinner("Analyzing text... This may take a moment."):
            
            try:
                # Make prediction
                prediction = model.predict([user_text])[0]
                probabilities = model.predict_proba([user_text])[0]
                
                # Format results
                class_names = ['Non-Depression', 'Depression']
                predicted_class = class_names[prediction]
                confidence = max(probabilities) * 100
                prob_depression = probabilities[1] * 100
                prob_non_depression = probabilities[0] * 100
                
                # Dark Glassmorphism Results Header
                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%); 
                           backdrop-filter: blur(30px); padding: 2rem; border-radius: 20px; margin: 2rem 0;
                           border: 1px solid rgba(255, 255, 255, 0.2);
                           box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);">
                    <h3 style="color: white; margin: 0; text-align: center; font-weight: 600; 
                               text-shadow: 0 0 15px rgba(102, 126, 234, 0.5); font-size: 1.5rem;">
                        🎯 AI Analysis Results
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Create modern results layout
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Dark Glassmorphism prediction cards
                    if predicted_class == "Depression":
                        st.markdown(f"""
                        <div style="background: rgba(255, 107, 107, 0.2); backdrop-filter: blur(20px);
                                   border: 1px solid rgba(255, 107, 107, 0.4); padding: 2rem; 
                                   border-radius: 20px; margin: 1rem 0;
                                   box-shadow: 0 15px 35px rgba(255, 107, 107, 0.3), 0 0 25px rgba(255, 107, 107, 0.2);">
                            <h4 style="margin: 0; font-weight: 600;
                                       background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
                                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
                                       text-shadow: 0 0 15px rgba(255, 107, 107, 0.5);">
                                ⚠️ Depression Indicators Detected
                            </h4>
                            <p style="color: #e2e8f0; margin: 1rem 0 0 0; font-size: 1.1rem;">
                                Confidence Level: <strong style="color: #ff6b6b; text-shadow: 0 0 10px rgba(255, 107, 107, 0.5);">{confidence:.1f}%</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: rgba(45, 206, 137, 0.2); backdrop-filter: blur(20px);
                                   border: 1px solid rgba(45, 206, 137, 0.4); padding: 2rem; 
                                   border-radius: 20px; margin: 1rem 0;
                                   box-shadow: 0 15px 35px rgba(45, 206, 137, 0.3), 0 0 25px rgba(45, 206, 137, 0.2);">
                            <h4 style="margin: 0; font-weight: 600;
                                       background: linear-gradient(135deg, #2dce89 0%, #4facfe 100%);
                                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
                                       text-shadow: 0 0 15px rgba(45, 206, 137, 0.5);">
                                ✅ No Depression Indicators
                            </h4>
                            <p style="color: #e2e8f0; margin: 1rem 0 0 0; font-size: 1.1rem;">
                                Confidence Level: <strong style="color: #2dce89; text-shadow: 0 0 10px rgba(45, 206, 137, 0.5);">{confidence:.1f}%</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Dark Glassmorphism probability breakdown
                    st.markdown("""
                    <div style="background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(20px);
                               padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.15);
                               box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);">
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <h4 style="margin: 0 0 1rem 0; font-weight: 600; color: #ffffff;
                               background: linear-gradient(135deg, #667eea 0%, #19a7ff 100%);
                               -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                        📊 Probability Breakdown
                    </h4>
                    """, unsafe_allow_html=True)
                    
                    # Modern progress bars with glassmorphism
                    st.markdown(f"**Depression:** {prob_depression:.1f}%")
                    st.progress(prob_depression / 100)
                    
                    st.markdown(f"**Non-Depression:** {prob_non_depression:.1f}%")
                    st.progress(prob_non_depression / 100)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Modern interpretation section
                if confidence >= 80:
                    confidence_text = "High Confidence"
                    confidence_icon = "🎯"
                    confidence_bg = "#e6f3ff"
                elif confidence >= 60:
                    confidence_text = "Moderate Confidence" 
                    confidence_icon = "⚡"
                    confidence_bg = "#fff8e1"
                else:
                    confidence_text = "Low Confidence"
                    confidence_icon = "�"
                    confidence_bg = "#ffeaa7"
                
                # Glassmorphism confidence indicator
                if confidence >= 80:
                    glass_bg = "rgba(79, 172, 254, 0.1)"
                    glass_border = "rgba(79, 172, 254, 0.3)"
                    gradient_text = "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
                elif confidence >= 60:
                    glass_bg = "rgba(255, 193, 7, 0.1)"
                    glass_border = "rgba(255, 193, 7, 0.3)"
                    gradient_text = "linear-gradient(135deg, #ffc107 0%, #ff8f00 100%)"
                else:
                    glass_bg = "rgba(255, 152, 0, 0.1)"
                    glass_border = "rgba(255, 152, 0, 0.3)"
                    gradient_text = "linear-gradient(135deg, #ff9800 0%, #ff5722 100%)"
                
                st.markdown(f"""
                <div style="background: {glass_bg}; backdrop-filter: blur(20px);
                           padding: 2rem; border-radius: 20px; margin: 2rem 0;
                           border: 1px solid {glass_border};
                           box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);">
                    <h4 style="margin: 0; font-weight: 600;
                               background: {gradient_text};
                               -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                        {confidence_icon} {confidence_text} ({confidence:.1f}%)
                    </h4>
                    <div style="margin-top: 1.5rem; color: #4a5568; line-height: 1.7;">
                        <p style="margin: 0.5rem 0;"><strong>AI Assessment:</strong> The model classifies this text as <strong>{predicted_class.lower()}</strong>-related</p>
                        <p style="margin: 0.5rem 0;"><strong>Important:</strong> This is an experimental research tool. Results should not influence medical decisions.</p>
                        <p style="margin: 0.5rem 0;"><strong>Next Steps:</strong> Always consult mental health professionals for proper assessment and support.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Glassmorphism crisis resources for depression detection
                if predicted_class == "Depression":
                    st.markdown("""
                    <div style="background: rgba(255, 107, 107, 0.12); backdrop-filter: blur(25px);
                               border: 1px solid rgba(255, 107, 107, 0.25); padding: 2.5rem; 
                               border-radius: 25px; margin: 2rem 0;
                               box-shadow: 0 15px 35px rgba(255, 107, 107, 0.15);">
                        <h4 style="margin: 0; text-align: center; font-weight: 600;
                                   background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%);
                                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                            💙 Support & Resources Available
                        </h4>
                        <div style="color: #555; margin-top: 1.5rem; text-align: center; line-height: 1.6;">
                            <p style="font-size: 1.1rem; margin: 0.5rem 0;"><strong>If you're struggling, you're not alone.</strong></p>
                            <p style="font-size: 1.05rem; margin: 0.5rem 0;">Professional help is available 24/7. Reaching out is a sign of strength.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Crisis resources in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        for i, (resource, contact) in enumerate(list(CRISIS_RESOURCES.items())[:2]):
                            st.markdown(f"**{resource}**")
                            st.code(contact)
                    with col2:
                        for resource, contact in list(CRISIS_RESOURCES.items())[2:]:
                            st.markdown(f"**{resource}**")
                            st.code(contact)
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.info("Please try again or contact support if the issue persists.")
    
    # Show disclaimers at the bottom
    show_disclaimers()
    
    # Glassmorphism Footer
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 50%, rgba(25, 167, 255, 0.9) 100%); 
               backdrop-filter: blur(25px); padding: 3rem 2rem; border-radius: 25px; text-align: center; margin-top: 3rem;
               border: 1px solid rgba(255, 255, 255, 0.3);
               box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1), 0 5px 15px rgba(0, 0, 0, 0.07);'>
        <div style='color: white;'>
            <h4 style='margin: 0; font-weight: 600; font-size: 1.4rem; text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                🧠 Mental Health AI Research Project
            </h4>
            <p style='margin: 1.5rem 0; opacity: 0.95; font-size: 1.1rem; text-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                Advanced NLP for Mental Health Insights | Built with Machine Learning & Compassion
            </p>
            <div style='background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); 
                        padding: 1.5rem; border-radius: 15px; margin-top: 1.5rem;
                        border: 1px solid rgba(255,255,255,0.25);'>
                <p style='margin: 0; font-size: 1rem; line-height: 1.6; text-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                    <strong>⚠️ Research Tool Only:</strong> Not for medical diagnosis or treatment decisions.<br>
                    Always consult qualified mental health professionals for support and assessment.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()