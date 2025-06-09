import streamlit as st
import joblib
import re
import string
from newspaper import Article
from deep_translator import GoogleTranslator
from langdetect import detect  # ✅ Language detection fix

# --- Page Configuration ---
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")

# --- Light Mode CSS ---
st.markdown("""
    <style>
    body {
        background-color: #f4f6f9;
    }
    .confidence-bar {
        height: 25px;
        background-color: #e0e0e0;
        border-radius: 25px;
        overflow: hidden;
        margin-top: 10px;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Text Preprocessing ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --- Load Models and Vectorizers ---
try:
    vectorizer = joblib.load("vectorizer.jb")              # For LR & RF
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.jb")  # For SVM
    vectorizer_nb = joblib.load("vectorizer_nb.jb")        # For Naive Bayes

    lr_model = joblib.load("lr_model.jb")
    rf_model = joblib.load("rf_model.jb")
    svm_model = joblib.load("svm_model.jb")
    nb_model = joblib.load("nb_model.jb")
except FileNotFoundError as e:
    st.error(f"❌ Required model or vectorizer file not found: {e}")
    st.stop()

# --- Title ---
st.markdown("<h1 style='text-align: center;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Paste a news article or enter a URL below.</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Input Mode Toggle ---
input_mode = st.radio("Choose Input Mode:", ("Manual Text", "Analyze from URL"))

if input_mode == "Manual Text":
    inputn = st.text_area("📄 Paste News Article", height=200, placeholder="Start typing or paste your article here...")
else:
    url = st.text_input("🌐 Enter a News Article URL")
    inputn = ""
    if url:
        try:
            article = Article(url)
            article.download()
            article.parse()
            inputn = article.text
            st.markdown("📰 **Extracted Article Text:**")
            st.info(inputn)
        except Exception as e:
            st.error(f"⚠️ Failed to fetch article: {e}")

# --- Language Detection + Translation ---
lang_option = st.checkbox("🌐 Translate to English (Auto-detect Language)", value=True)

# -----Bert logic add Start here------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned BERT model and tokenizer
tokenizer_bert = AutoTokenizer.from_pretrained("./bert_fakenews")
bert_model = AutoModelForSequenceClassification.from_pretrained("./bert_fakenews")
bert_model.eval()


def predict_with_bert(text):
    inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_label].item()
    return predicted_label, confidence

# def predict_with_bert(text):
#     inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#         logits = outputs.logits
#         probabilities = torch.softmax(logits, dim=1)[0]  # 1D tensor
#     return probabilities.cpu().numpy()  # Returns array like: [fake_prob, real_prob]

# -----Bert logic add End here------

# --- Prediction ---
if st.button("🔍 Analyze News"):
    if inputn.strip():
        original_text = inputn
        if lang_option:
            try:
                detected_lang = detect(inputn)
                st.markdown(f"🌐 Detected Language: **{detected_lang.upper()}**")

                if detected_lang.lower() != "en":
                    translated = GoogleTranslator(source='auto', target='en').translate(inputn)
                    st.markdown("🈯 **Translated to English:**")
                    st.info(translated)
                    inputn = translated
                else:
                    st.success("✅ No translation needed. The text is already in English.")
            except Exception as e:
                st.warning(f"⚠️ Language detection or translation failed: {e}")
                inputn = original_text

        cleaned = clean_text(inputn)
        if len(cleaned.split()) < 15:
            st.warning("⚠️ Please enter at least 15 words for better accuracy.")
        else:
            # Vectorize input
            vec_input_lr_rf = vectorizer.transform([cleaned])         # For LR & RF
            vec_input_svm = tfidf_vectorizer.transform([cleaned])     # For SVM
            vec_input_nb = vectorizer_nb.transform([cleaned])         # For Naive Bayes

            # --- Predictions ---
            lr_pred = lr_model.predict(vec_input_lr_rf)[0]
            lr_conf = lr_model.predict_proba(vec_input_lr_rf)[0]

            rf_pred = rf_model.predict(vec_input_lr_rf)[0]
            rf_conf = rf_model.predict_proba(vec_input_lr_rf)[0]

            svm_pred = svm_model.predict(vec_input_svm)[0]

            nb_pred = nb_model.predict(vec_input_nb)[0]
            nb_conf = nb_model.predict_proba(vec_input_nb)[0]

            # --- Display Results ---
            st.markdown("### 🔍 Predictions by Models")

            st.markdown(f"**Logistic Regression:** {'🟢 Real' if lr_pred == 1 or lr_pred == 'Real' else '🔴 Fake'}")
            st.markdown(f"🟢 Real: `{lr_conf[1]*100:.2f}%` | 🔴 Fake: `{lr_conf[0]*100:.2f}%`")
            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown(f"**Random Forest:** {'🟢 Real' if rf_pred == 1 or rf_pred == 'Real' else '🔴 Fake'}")
            st.markdown(f"🟢 Real: `{rf_conf[1]*100:.2f}%` | 🔴 Fake: `{rf_conf[0]*100:.2f}%`")
            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown(f"**Support Vector Machine:** {'🟢 Real' if svm_pred == 1 or svm_pred == 'Real' else '🔴 Fake'}")
            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown(f"**Naive Bayes:** {'🟢 Real' if nb_pred == 1 or nb_pred == 'Real' else '🔴 Fake'}")
            st.markdown(f"🟢 Real: `{nb_conf[1]*100:.2f}%` | 🔴 Fake: `{nb_conf[0]*100:.2f}%`")
            
            # Bert logic add here----
            # --- After Naive Bayes ---
            st.markdown("<hr>", unsafe_allow_html=True)

            # --- BERT Prediction ---
            bert_pred, bert_conf = predict_with_bert(cleaned)
            st.markdown(f"**BERT (Transformer):** {'🟢 Real' if bert_pred == 1 else '🔴 Fake'}")
            st.markdown(f"🧠 Confidence: `{bert_conf * 100:.2f}%`")
            # bert_probs = predict_with_bert(cleaned)
            # bert_fake_pct = bert_probs[0] * 100
            # bert_real_pct = bert_probs[1] * 100
            # bert_pred = "Real" if bert_real_pct > bert_fake_pct else "Fake"

            # st.markdown(f"**BERT (Transformer):** {'🟢 Real' if bert_pred == 'Real' else '🔴 Fake'}")
            # st.markdown(f"🟢 Real: `{bert_real_pct:.2f}%` | 🔴 Fake: `{bert_fake_pct:.2f}%`")
  

    else:
        st.warning("⚠️ Please enter a news article or valid URL.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>Made By Irteja using Streamlit |</br> Updated By Pankaj Mahanta | © 2025</p>", unsafe_allow_html=True)
