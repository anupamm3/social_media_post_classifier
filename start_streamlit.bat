@echo off
echo Starting Streamlit Mental Health Tweet Classifier Demo...
cd /d "d:\Project\mental_health_tweet"
echo. | streamlit run simple_demo.py --server.port 8501 --server.address localhost
pause