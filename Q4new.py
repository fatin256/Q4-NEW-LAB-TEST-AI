import streamlit as st
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from PyPDF2 import PdfReader

# Only download punkt (no punkt_tab)
nltk.download("punkt", quiet=True)

st.set_page_config(page_title="Q4: Text Chunking (NLTK)", layout="wide")
st.title("Q4: Text Chunking using NLTK Sentence Tokenizer")

st.write(
    """
This web app follows the required steps:
1. Import PDF using PdfReader (PyPDF2)
2. Extract text from the uploaded PDF
3. Split text into sentences and display indices 58–68
4. Perform semantic sentence chunking using NLTK
"""
)

# ----------------------------
# Step 1 & 2: PDF import + extraction
# ----------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text_pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_pages.append(page_text)
    return "\n".join(text_pages)

uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_pdf is None:
    st.info("Please upload a PDF file to begin.")
else:
    extracted_text = extract_text_from_pdf(uploaded_pdf)

    if not extracted_text.strip():
        st.error("No text extracted. This PDF may be scanned (image-based).")
    else:
        st.subheader("Step 2: Extracted Text (Preview)")
        st.text_area("Preview (first 2000 characters)", extracted_text[:2000], height=200)

        # ----------------------------
        # Step 3: Sentence splitting (SAFE)
        # ----------------------------
        st.subheader("Step 3: Sentence Splitting")

        tokenizer = PunktSentenceTokenizer()
        sentences = tokenizer.tokenize(extracted_text)

        st.write(f"Total sentences found: **{len(sentences)}**")

        start_i, end_i = 58, 68

        if len(sentences) <= start_i:
            st.warning(
                f"Not enough sentences. Found only {len(sentences)} sentences."
            )
        else:
            sample = []
            for i in range(start_i, min(end_i + 1, len(sentences))):
                sample.append({"Index": i, "Sentence": sentences[i]})

            st.table(sample)

            # ----------------------------
            # Step 4: Semantic sentence chunking
            # ----------------------------
            st.subheader("Step 4: Semantic Sentence Chunks (NLTK)")
            for row in sample:
                st.markdown(f"**[{row['Index']}]** {row['Sentence']}")
