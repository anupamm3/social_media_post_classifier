"""
Simple demo script for the Mental Health Tweet Classifier.
This is a command-line interface to test the trained model.
"""

import sys
from pathlib import Path
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def load_model():
    """Load the trained baseline model."""
    try:
        model_path = Path("models/baseline/logistic_model.joblib")
        vectorizer_path = Path("models/baseline/tfidf_vectorizer.joblib")
        
        if not model_path.exists() or not vectorizer_path.exists():
            print("❌ Model files not found. Please train the model first.")
            print("   Run the notebook: notebooks/01_explore.ipynb")
            return None, None
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        print("✅ Model loaded successfully!")
        return model, vectorizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def predict_text(model, vectorizer, text):
    """Make a prediction on input text."""
    try:
        # Transform text using vectorizer
        text_tfidf = vectorizer.transform([text])
        
        # Get prediction and probability
        pred = model.predict(text_tfidf)[0]
        prob = model.predict_proba(text_tfidf)[0]
        
        # Format results
        label_name = "Depression" if pred == 1 else "Non-Depression"
        confidence = max(prob) * 100
        prob_depression = prob[1] * 100
        prob_non_depression = prob[0] * 100
        
        return {
            'prediction': label_name,
            'confidence': confidence,
            'prob_depression': prob_depression,
            'prob_non_depression': prob_non_depression
        }
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None

def show_crisis_resources():
    """Display crisis resources."""
    print("\n🆘 CRISIS RESOURCES:")
    print("   If you or someone you know needs help:")
    print("   • US National Suicide Prevention Lifeline: 988")
    print("   • US Crisis Text Line: Text HOME to 741741")
    print("   • UK Samaritans: 116 123")
    print("   • International: https://www.iasp.info/resources/Crisis_Centres/")

def show_disclaimer():
    """Display important disclaimers."""
    print("\n⚠️  IMPORTANT DISCLAIMER:")
    print("   This tool is for RESEARCH and EDUCATIONAL purposes ONLY.")
    print("   • NOT for clinical diagnosis or medical decisions")
    print("   • NOT for crisis intervention or emergency response")
    print("   • Always consult qualified mental health professionals")
    print("   • Results may contain errors and biases")

def main():
    """Main demo function."""
    print("🧠 Mental Health Tweet Classifier - Demo")
    print("="*50)
    
    # Show important disclaimers
    show_disclaimer()
    show_crisis_resources()
    
    # Load model
    print(f"\n📊 Loading model...")
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        return
    
    print(f"\n🎯 Model ready! Type 'quit' to exit, 'help' for commands.")
    print(f"💡 Try some example texts or enter your own.")
    
    # Example texts
    examples = [
        "I feel so sad and hopeless, nothing matters anymore",
        "Having a great day, feeling amazing!",
        "Everything is falling apart, I can't take it anymore",
        "Just finished a workout, feeling energized"
    ]
    
    while True:
        print(f"\n" + "-"*50)
        user_input = input("\n📝 Enter text to analyze (or command): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        elif user_input.lower() in ['help', 'h']:
            print("\n💡 Available commands:")
            print("   • help/h - Show this help")
            print("   • examples/e - Show example predictions")
            print("   • crisis/c - Show crisis resources")
            print("   • quit/exit/q - Exit the demo")
            print("   • Or just type any text to analyze")
            continue
        elif user_input.lower() in ['examples', 'e']:
            print(f"\n🧪 Example predictions:")
            for i, example in enumerate(examples, 1):
                result = predict_text(model, vectorizer, example)
                if result:
                    print(f"\n{i}. \"{example[:60]}{'...' if len(example) > 60 else ''}\"")
                    print(f"   → {result['prediction']} ({result['confidence']:.1f}% confidence)")
                    print(f"   → Depression: {result['prob_depression']:.1f}% | Non-Depression: {result['prob_non_depression']:.1f}%")
            continue
        elif user_input.lower() in ['crisis', 'c']:
            show_crisis_resources()
            continue
        elif not user_input:
            print("❓ Please enter some text to analyze.")
            continue
        
        # Make prediction
        result = predict_text(model, vectorizer, user_input)
        
        if result:
            print(f"\n📊 ANALYSIS RESULTS:")
            print(f"   Text: \"{user_input[:80]}{'...' if len(user_input) > 80 else ''}\"")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.1f}%")
            print(f"   Probabilities:")
            print(f"     • Depression: {result['prob_depression']:.1f}%")
            print(f"     • Non-Depression: {result['prob_non_depression']:.1f}%")
            
            # Show warning for depression predictions
            if result['prediction'] == 'Depression' and result['confidence'] > 70:
                print(f"\n⚠️  High depression likelihood detected.")
                print(f"   Remember: This is NOT a medical diagnosis.")
                print(f"   If you're struggling, please seek professional help.")
                show_crisis_resources()
        
        print(f"\n💡 Enter another text or type 'help' for commands.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n👋 Demo stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"Please check your installation and try again.")