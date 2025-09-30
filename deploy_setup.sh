#!/bin/bash
# Quick deployment setup script

echo "🚀 Mental Health AI - Quick Deployment Setup"
echo "=========================================="

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📁 Initializing git repository..."
    git init
    git branch -M main
fi

# Check if model files exist
if [ ! -f "models/baseline/logistic_model.joblib" ]; then
    echo "⚠️  WARNING: Model files not found!"
    echo "   Please run the Jupyter notebook to train the model first."
    echo "   Expected files:"
    echo "   - models/baseline/logistic_model.joblib"
    echo "   - models/baseline/tfidf_vectorizer.joblib"
    exit 1
fi

# Add all files
echo "📦 Adding files to git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Deployment ready - Mental Health AI Research App"

echo ""
echo "✅ Repository is ready for deployment!"
echo ""
echo "🌟 Next Steps:"
echo "1. Push to GitHub:"
echo "   git remote add origin https://github.com/YOURUSERNAME/mental_health_tweet.git"
echo "   git push -u origin main"
echo ""
echo "2. Deploy on Streamlit Cloud:"
echo "   → Go to share.streamlit.io"
echo "   → Connect your GitHub repository"
echo "   → Set main file: simple_demo.py"
echo "   → Click Deploy!"
echo ""
echo "3. Future updates:"
echo "   → Just run: git add . && git commit -m 'Update' && git push"
echo "   → Your app will auto-update in 2-5 minutes!"
echo ""
echo "📖 See DEPLOYMENT_GUIDE.md for detailed instructions."