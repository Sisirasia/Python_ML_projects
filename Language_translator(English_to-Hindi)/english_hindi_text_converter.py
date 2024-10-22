import streamlit as st
from deep_translator import GoogleTranslator
from fpdf import FPDF
from io import BytesIO
import os

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

    # Provide the full path to the Mangal font
    font_path = os.path.join(os.path.dirname(__file__), "MANGAL.TTF")  # Adjust the path if necessary

    # Check if the font file exists
    if not os.path.exists(font_path):
        raise FileNotFoundError("The MANGAL.TTF font file was not found. Ensure the font is available in the specified path.")

    # Set the font to a Unicode-compatible font (Mangal.ttf)
    pdf.add_font("Devanagari", '', font_path, uni=True)
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
    pdf_output.write(pdf.output(dest='S').encode('latin1'))  # Use 'S' for string output and encode to binary format
    pdf_output.seek(0)
    return pdf_output

def main():
    st.set_page_config(page_title="English to Hindi PDF Translator")
    st.header("Enter English Text and Convert it to Hindi")

    # Get English text input
    english_text = st.text_area("Enter the English text you want to translate")

    if english_text:
        # Translate the text into Hindi
        with st.spinner("Translating to Hindi..."):
            hindi_text = translate_text(english_text)
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
