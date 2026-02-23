import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt

# -------------------- OPTIONAL: Set Tesseract Path for Windows --------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------- API KEY SETUP --------------------
load_dotenv("apimed.env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("🚨 GEMINI_API_KEY not found. Please add it to apikey.env")
    st.stop()

# -------------------- OCR FUNCTIONS --------------------
def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image, lang='eng')
    return text

def extract_text_from_pdf(pdf_path):
    try:
        from pdf2image import convert_from_path
    except ImportError:
        st.error("📌 pdf2image not installed. Install it to process PDFs.")
        return ""
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang='eng') + "\n"
    return text

# -------------------- Medical Problem Analysis --------------------
def explain_medical_report(text, tone_choice, language_choice):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=api_key
    )

    chunks = [Document(page_content=text[i:i+20000]) for i in range(0, len(text), 20000)]

    prompt_template = f"""
    You are an expert medical assistant.
    A patient has uploaded this lab report:

    {{text}}

    Task:
    1. Identify the main medical problem from the report.
    2. List the lab values relevant to this problem.
    3. Suggest a clear diet structure (foods, fruits, meals) the patient should follow.
    4. Compare patient's current values with normal/recommended values.
    5. Use a {tone_choice} tone and explain in {language_choice}.
    6. Output only analysis, diet plan, and relevant lab value summary. No other information.

    Output:
    """

    prompt = PromptTemplate.from_template(prompt_template)
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    result = chain.invoke({"input_documents": chunks})
    return result['output_text']

# -------------------- STREAMLIT APP --------------------
def main():
    st.set_page_config(page_title="Medical Report Interpreter", page_icon="🩺", layout="wide")
    st.markdown("<h1 style='text-align:center;'>🩺 Medical Report Interpreter and Diet Suggestor</h1>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "📂 Upload your scanned report(s) (PDF or Image)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    tone_choice = st.selectbox(
        "🎭 Select explanation tone",
        ["simple", "friendly", "professional", "empathetic"],
        index=0
    )

    language_choice = st.selectbox(
        "🌐 Select language",
        ["English", "Hindi", "Telugu", "Spanish", "French", "German"],
        index=0
    )

    if uploaded_files and st.button("✨ Generate Analysis & Diet Plan"):
        with st.spinner("🤖 Gemini is analyzing the report(s)..."):
            try:
                combined_text = ""

                for uploaded_file in uploaded_files:
                    temp_file_path = None
                    if uploaded_file.type == "application/pdf":
                        temp_dir = tempfile.mkdtemp()
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        combined_text += extract_text_from_pdf(temp_file_path) + "\n"
                    else:
                        combined_text += extract_text_from_image(uploaded_file) + "\n"

                if not combined_text.strip():
                    st.warning("⚠️ No text could be extracted from the uploaded file(s).")
                    return

                # Generate analysis & diet plan
                explanation = explain_medical_report(combined_text, tone_choice, language_choice)
                st.markdown("### ✅ Analysis & Diet Plan")
                st.markdown(
                    f"<div style='background:#000000;padding:15px;border-left:5px solid #1a73e8;border-radius:8px;'>{explanation}</div>",
                    unsafe_allow_html=True
                )

                # -------------------- Compact Lab Values Graph --------------------
                st.markdown("### 📊 Lab Values: Current vs Recommended")
                labels = ["Hemoglobin", "WBC"]
                current_values = [10.2, 12000]
                recommended_values = [13.5, 11000]

                fig, ax = plt.subplots(figsize=(3.5,2))
                x = range(len(labels))
                bar_width = 0.25

                ax.bar(x, recommended_values, width=bar_width, label='Recommended', color='green', align='center')
                ax.bar([i + bar_width for i in x], current_values, width=bar_width, label='Current', color='red', align='center')

                ax.set_xticks([i + bar_width/2 for i in x])
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_ylabel("Values", fontsize=8)
                ax.set_title("Lab Values Comparison", fontsize=9)
                ax.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
