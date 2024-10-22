import streamlit as st
from PyPDF2 import PdfReader
from deep_translator import GoogleTranslator
from fpdf import FPDF
from io import BytesIO
 
# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
 
# Function to chunk text into smaller parts for translation
def split_text_into_chunks(text, max_length=5000):
    chunks = []
    while len(text) > max_length:
        split_index = text[:max_length].rfind(' ')  # Split at the last space within max_length
        chunks.append(text[:split_index])
        text = text[split_index:]
    chunks.append(text)
    return chunks
 
# Function to translate text from English to Hindi
def translate_text(text):
    translator = GoogleTranslator(source='en', target='hi')
    text_chunks = split_text_into_chunks(text)
    translated_text = ""
    for chunk in text_chunks:
        translated_text += translator.translate(chunk)
    return translated_text
 
# Function to create a PDF with translated Hindi text
def create_hindi_pdf(translated_text):
    pdf = FPDF()
    pdf.add_page()
 
    # Set the font to a Unicode-compatible font (Mangal.ttf)
    pdf.add_font("Devanagari", '', 'MANGAL.TTF', uni=True)
    pdf.set_font("Devanagari", size=12)
 
    # Set up page margins and width to handle text wrapping better
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)
    page_width = pdf.w - pdf.l_margin - pdf.r_margin  # Calculate available page width
 
    # Split the translated text into lines and add to the PDF
    lines = translated_text.split('\n')
    for line in lines:
        pdf.multi_cell(page_width, 10, line)
 
    # Save PDF to a BytesIO object to allow downloading
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output
 
def main():
    st.set_page_config(page_title="English to Hindi PDF Translator")
    st.header("Upload an English PDF and Convert it to Hindi")
 
    # Upload PDF
    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=False, type=["pdf"])
    if pdf_docs is not None:
        # Extract text from PDF
        raw_text = get_pdf_text([pdf_docs])
        # Display extracted text
        st.subheader("Extracted English Text:")
        st.write(raw_text)
        # Translate the text into Hindi
        with st.spinner("Translating to Hindi..."):
            hindi_text = translate_text(raw_text)
            st.success("Translation Complete!")
        # Display translated Hindi text
        st.subheader("Translated Hindi Text:")
        st.write(hindi_text)
        # Create Hindi PDF
        hindi_pdf = create_hindi_pdf(hindi_text)
 
        # Provide download option for the translated PDF
        st.download_button(label="Download Translated PDF",
                           data=hindi_pdf,
                           file_name="translated_hindi.pdf",
                           mime="application/pdf")
 
if __name__ == "__main__":
    main()
 

                