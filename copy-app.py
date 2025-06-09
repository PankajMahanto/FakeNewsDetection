import streamlit as st
import joblib
import re
import string
from newspaper import Article
from deep_translator import GoogleTranslator
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Page Setup ---
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    body { background-color: #f4f6f9; }
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

# --- Text Cleaner ---
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

# --- Load vectorizers + classical models ---
try:
    vectorizer = joblib.load("vectorizer.jb")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.jb")
    vectorizer_nb = joblib.load("vectorizer_nb.jb")

    lr_model = joblib.load("lr_model.jb")
    rf_model = joblib.load("rf_model.jb")
    svm_model = joblib.load("svm_model.jb")
    nb_model = joblib.load("nb_model.jb")
except FileNotFoundError as e:
    st.error(f"❌ Required model or vectorizer file not found: {e}")
    st.stop()

# --- Load BERT Model & Tokenizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_bert = AutoTokenizer.from_pretrained("./bert_fakenews")
bert_model = AutoModelForSequenceClassification.from_pretrained("./bert_fakenews").to(device)
bert_model.eval()

def predict_with_bert(text):
    inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    return probs.cpu().numpy()  # [Fake_prob, Real_prob]

# --- UI Starts Here ---
st.markdown("<h1 style='text-align: center;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Paste a news article or enter a URL.</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

input_mode = st.radio("Choose Input Mode:", ("Manual Text", "Analyze from URL"))

if input_mode == "Manual Text":
    inputn = st.text_area("📄 Paste News Article", height=200)
else:
    url = st.text_input("🌐 Enter News Article URL")
    inputn = ""
    if url:
        try:
            article = Article(url)
            article.download()
            article.parse()
            inputn = article.text
            st.info(inputn)
        except Exception as e:
            st.error(f"❌ Failed to fetch article: {e}")

lang_option = st.checkbox("🌐 Translate to English (Auto-detect)", value=True)

if st.button("🔍 Analyze News"):
    if inputn.strip():
        original = inputn
        if lang_option:
            try:
                lang = detect(inputn)
                st.markdown(f"🌐 Detected Language: **{lang.upper()}**")
                if lang != "en":
                    inputn = GoogleTranslator(source='auto', target='en').translate(inputn)
                    st.info(inputn)
                else:
                    st.success("✅ Already English.")
            except Exception as e:
                st.warning(f"⚠️ Translation error: {e}")
                inputn = original

        cleaned = clean_text(inputn)
        if len(cleaned.split()) < 15:
            st.warning("⚠️ Please enter at least 15 words.")
        else:
            # Classical Models
            vec_lr_rf = vectorizer.transform([cleaned])
            vec_svm = tfidf_vectorizer.transform([cleaned])
            vec_nb = vectorizer_nb.transform([cleaned])

            lr_pred = lr_model.predict(vec_lr_rf)[0]
            lr_conf = lr_model.predict_proba(vec_lr_rf)[0]

            rf_pred = rf_model.predict(vec_lr_rf)[0]
            rf_conf = rf_model.predict_proba(vec_lr_rf)[0]

            svm_pred = svm_model.predict(vec_svm)[0]
            nb_pred = nb_model.predict(vec_nb)[0]
            nb_conf = nb_model.predict_proba(vec_nb)[0]

            # Display classical results
            st.markdown("### 🧠 Classic ML Predictions")
            st.markdown(f"**Logistic Regression:** {'🟢 Real' if lr_pred == 1 else '🔴 Fake'}")
            st.markdown(f"🟢 Real: `{lr_conf[1]*100:.2f}%` | 🔴 Fake: `{lr_conf[0]*100:.2f}%`")
            st.markdown("---")

            st.markdown(f"**Random Forest:** {'🟢 Real' if rf_pred == 1 else '🔴 Fake'}")
            st.markdown(f"🟢 Real: `{rf_conf[1]*100:.2f}%` | 🔴 Fake: `{rf_conf[0]*100:.2f}%`")
            st.markdown("---")

            st.markdown(f"**SVM:** {'🟢 Real' if svm_pred == 1 else '🔴 Fake'}")
            st.markdown("---")

            st.markdown(f"**Naive Bayes:** {'🟢 Real' if nb_pred == 1 else '🔴 Fake'}")
            st.markdown(f"🟢 Real: `{nb_conf[1]*100:.2f}%` | 🔴 Fake: `{nb_conf[0]*100:.2f}%`")
            st.markdown("---")

            # 🔹 BERT Prediction
            bert_probs = predict_with_bert(cleaned)
            fake_pct = bert_probs[0] * 100
            real_pct = bert_probs[1] * 100
            bert_label = "Real" if real_pct > fake_pct else "Fake"

            st.markdown("### 🤖 BERT (Transformer) Prediction")
            st.markdown(f"**Prediction:** {'🟢 Real' if bert_label == 'Real' else '🔴 Fake'}")
            st.markdown(f"🟢 Real: `{real_pct:.2f}%` | 🔴 Fake: `{fake_pct:.2f}%`")
    else:
        st.warning("⚠️ Please provide article text or valid URL.")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>🔎 Made by Irteja, Mostak, Sagor | Updated by Pankaj | © 2025</p>", unsafe_allow_html=True)
