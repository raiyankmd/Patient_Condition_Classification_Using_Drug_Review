import streamlit as st
import pickle
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# PAGE CONFIG
 
st.set_page_config(
    page_title="Patient Condition Classifier",
    page_icon="🧠",
    layout="wide"
)

# Load model with caching

@st.cache_resource
def load_model():
    with open('patient_condition_pipeline.pkl', 'rb') as f:
        return pickle.load(f)


def clean_text(text):
    
    text = str(text).lower()
    
    # Remove contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords & lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    text = " ".join(tokens)
    
    return text

pipeline = load_model()




# CUSTOM STYLING

st.markdown("""
<style>
.big-font {
    font-size:22px !important;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #f0f8ff;
    border: 2px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# HEADER 
st.title("🧠 Drug Review–Based Patient Condition Classification")

st.markdown(
"""
This AI system predicts a patient's medical condition based on a drug review using Natural Language Processing and Machine Learning.
"""
)

col1, col2 = st.columns(2)

with col1:

    review = st.text_area(
        "📝 Enter Drug Review",
        height=250,
        placeholder="Example: My blood pressure was extremely high, but after taking this medication it is now under control."
    )

    predict_btn = st.button("🔍 Predict Condition")

with col2:

    st.info(
        """
        ### 💡 Tips for Best Results
        - Write a detailed review
        - Mention symptoms or effects
        - Avoid very short text
        """
    )

    st.markdown("### 📌 Supported Conditions")
    st.write("• Depression")
    st.write("• Diabetes Type 2")
    st.write("• High Blood Pressure")

# PREDICTION 
if predict_btn:

    # Error handling
    if review.strip() == "":
        st.warning("⚠️ Please enter a review before predicting.")
    
    elif len(review.split()) < 5:
        st.warning("⚠️ Review is too short. Please enter more details.")
    
    else:

        # Prediction
        condition  = pipeline.predict([review])[0]
        probs = pipeline.predict_proba([review])[0]

        # condition = le.inverse_transform([pred])[0]
        confidence = np.max(probs) * 100

        if condition != "":
            
            # RESULT DISPLAY 
            st.markdown("---")
            st.markdown("## 🧾 Prediction Result")

            st.markdown(
                f"""
                <div>
                <h3>Predicted Condition: {condition}</h3>
                <p class="big-font">Confidence Score: {confidence:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.warning("Prediction failed")

        st.markdown("---")

        if len(probs) > 0:
            # PROBABILITY BREAKDOWN 
            st.markdown("### 📊 Probability Distribution")

            prob_dict = {
                pipeline.classes_[i]: probs[i]*100
                for i in range(len(probs))
            }

            st.bar_chart(prob_dict)

# SIDEBAR
st.sidebar.title("📘 About This Project")

st.sidebar.write("""
**Project:** Patient Condition Classification  
**Approach:** NLP + Machine Learning  
**Model:** Logistic Regression  
**Feature Extraction:** TF-IDF  
""")

st.sidebar.markdown("---")

st.sidebar.write("### 👨‍⚕️ How It Works")

st.sidebar.write("""
1. User enters drug review  
2. Text converted to TF-IDF features  
3. Model predicts condition  
4. Confidence score displayed  
""")

st.sidebar.markdown("---")

st.sidebar.write("### ⚠️ Disclaimer")
st.sidebar.write(
"This tool is for educational purposes only and not a medical diagnosis system."
)